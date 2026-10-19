from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from thsolver import Solver
from thsolver.tracker import AverageTracker
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuoct.config.schema import DataConfig, ModelConfig, validate_solver_config
from neuoct.data.dataset import ShapeNetMultiViewDataset, collate_samples
from neuoct.data.preprocess import DeviceBatchPreprocessor
from neuoct.data.sample import PreparedBatch
from neuoct.data.sampler import ObjectDistributedInfSampler
from neuoct.geometry.octree_adapter import build_input_features
from neuoct.geometry.vertex_index import VertexIndex
from neuoct.losses import IntersectionObjective, IntersectionTarget, ObjectiveWeights
from neuoct.losses.image import color_detail_weights_for_pairs, normal_detail_weights_for_pairs
from neuoct.metrics import (
    color_detail_metrics,
    geometry_noise_metrics,
    hit_classification_metrics,
    hit_topology_metrics,
    image_psnr,
    masked_psnr,
    normal_detail_metrics,
)
from neuoct.models import NeuralIntersectionModel
from neuoct.models.vertex_features import VertexFeatureField
from neuoct.rendering import SparseRenderer, generate_orbit_views, save_render_output

from .features import load_feature_export, save_feature_export
from .run_metadata import write_run_metadata


class NeuOctSolver(Solver):
    """Thin THSolver adapter for the refactored non-VAE pipeline."""

    def __init__(self, flags, is_master: bool = True):
        validate_solver_config(flags)
        super().__init__(flags, is_master)
        loss_flags = flags.LOSS
        if bool(getattr(flags.SOLVER, "tf32", False)):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.objective = IntersectionObjective(
            ObjectiveWeights(
                ssim_normal=float(getattr(loss_flags, "ssim_normal", 0.3)),
                ssim_color=float(getattr(loss_flags, "ssim_color", 1.0)),
                smoothness=float(getattr(loss_flags, "smoothness", 0.2)),
                constancy=float(getattr(loss_flags, "constancy", 0.1)),
                normal_gradient=float(getattr(loss_flags, "normal_gradient", 0.0)),
                normal_multiscale=float(
                    getattr(loss_flags, "normal_multiscale", 0.0)
                ),
                color_gradient=float(getattr(loss_flags, "color_gradient", 0.0)),
                color_laplacian=float(
                    getattr(loss_flags, "color_laplacian", 0.0)
                ),
                hit_hole=float(getattr(loss_flags, "hit_hole", 0.0)),
                hit_floater=float(getattr(loss_flags, "hit_floater", 0.0)),
                hit_floater_hard=float(
                    getattr(loss_flags, "hit_floater_hard", 0.0)
                ),
                hit_boundary=float(getattr(loss_flags, "hit_boundary", 0.0)),
                hit_boundary_f1=float(getattr(loss_flags, "hit_boundary_f1", 0.0)),
                hit_tversky=float(getattr(loss_flags, "hit_tversky", 0.0)),
                ray_distance=float(getattr(loss_flags, "ray_distance", 0.0)),
            ),
            positive_geometry=bool(getattr(loss_flags, "positive_geometry", False)),
            hit_loss_mode=str(getattr(loss_flags, "hit_loss_mode", "bce")),
            focal_gamma=float(getattr(loss_flags, "focal_gamma", 2.0)),
            hit_positive_weight=float(getattr(loss_flags, "hit_positive_weight", 1.0)),
            hit_term_weight=float(getattr(loss_flags, "hit_term_weight", 1.0)),
            normal_cosine_weight=float(getattr(loss_flags, "normal_cosine_weight", 0.0)),
            stage_weight_mode=str(getattr(loss_flags, "stage_weight_mode", "legacy")),
            final_stage_weight=float(getattr(loss_flags, "final_stage_weight", 0.5)),
            masked_normal_ssim=bool(getattr(loss_flags, "masked_normal_ssim", False)),
            masked_color_ssim=bool(getattr(loss_flags, "masked_color_ssim", False)),
            normal_detail_scale=float(getattr(loss_flags, "normal_detail_scale", 0.0)),
            color_detail_scale=float(getattr(loss_flags, "color_detail_scale", 0.0)),
            color_loss_mode=str(getattr(loss_flags, "color_loss_mode", "relative_mse")),
            color_weight=float(getattr(loss_flags, "color_weight", 2.0)),
            hit_floater_hard_fraction=float(
                getattr(loss_flags, "hit_floater_hard_fraction", 0.001)
            ),
            hit_boundary_f1_tolerance=int(
                getattr(loss_flags, "hit_boundary_f1_tolerance", 1)
            ),
            hit_tversky_alpha=float(getattr(loss_flags, "hit_tversky_alpha", 0.3)),
            hit_tversky_beta=float(getattr(loss_flags, "hit_tversky_beta", 0.7)),
            hit_tversky_gamma=float(getattr(loss_flags, "hit_tversky_gamma", 1.0)),
        ).to(torch.device("cuda", self.device))
        self.detail_weight_scale = float(getattr(loss_flags, "detail_weight_scale", 0.0))
        self.color_detail_weight_scale = float(
            getattr(loss_flags, "color_detail_weight_scale", 0.0)
        )
        self.color_rarity_weight_scale = float(
            getattr(loss_flags, "color_rarity_weight_scale", 0.0)
        )
        self.parameter_l2 = float(getattr(loss_flags, "parameter_l2", 0.0))
        self.parameter_l1 = float(getattr(loss_flags, "parameter_l1", 0.0))
        self.validate_each_checkpoint = bool(
            getattr(flags.SOLVER, "validate_each_checkpoint", False)
        )
        self.checkpoint_validation_batches = int(
            getattr(flags.SOLVER, "checkpoint_validation_batches", -1)
        )
        self._best_checkpoint_probe_val: float | None = None
        self._preprocessors: dict[int, DeviceBatchPreprocessor] = {}
        self._resume_iteration = 0

    @staticmethod
    def _atomic_torch_save(value, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        torch.save(value, temporary)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json_save(value: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    @staticmethod
    def _freeze_batch_norm_statistics(module: torch.nn.Module) -> int:
        count = 0
        for child in module.modules():
            if isinstance(child, _BatchNorm):
                child.eval()
                count += 1
        return count

    def _model_state_dict(self):
        return self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()

    def _solver_checkpoint(self, model_dict, *, epoch: int, iteration: int | None = None):
        checkpoint = {
            "model_dict": model_dict,
            "epoch": int(epoch),
            "optimizer_dict": self.optimizer.state_dict(),
            "scheduler_dict": self.scheduler.state_dict(),
            "scaler_dict": self.scaler.state_dict(),
            "world_size": int(self.world_size),
            "best_val": self.best_val,
        }
        if iteration is not None:
            checkpoint["iteration"] = int(iteration)
        return checkpoint

    def get_model(self, flags):
        model = ModelConfig.from_flags(flags)
        return NeuralIntersectionModel(
            model.input_channels,
            model.query_depth,
            renderer_hidden_dim=model.renderer_hidden_dim,
            renderer_iterations=model.renderer_iterations,
            segment_position=model.segment_position,
            voxel_feature_dim=model.voxel_feature_dim,
            color_skip_weight=model.color_skip_weight,
            activation_checkpointing=model.activation_checkpointing,
            material_fp32=model.material_fp32,
            norm_type=model.norm_type,
            norm_groups=model.norm_groups,
            sampling_mode=model.sampling_mode,
        )

    def get_dataset(self, flags):
        return ShapeNetMultiViewDataset(DataConfig.from_flags(flags)), collate_samples

    def get_dataloader(self, flags):
        dataset, collate_fn = self.get_dataset(flags)
        if self.world_size > 1:
            sampler = ObjectDistributedInfSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=bool(flags.shuffle),
            )
        else:
            from thsolver.sampler import InfSampler

            sampler = InfSampler(dataset, shuffle=bool(flags.shuffle))
        return DataLoader(
            dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=flags.pin_memory,
            persistent_workers=bool(flags.num_workers > 0),
            prefetch_factor=2 if flags.num_workers > 0 else None,
        )

    def configure_log(self, set_writer=True):
        super().configure_log(set_writer=set_writer)
        if self.is_master:
            write_run_metadata(
                Path(self.logdir) / "run.json",
                flags=self.FLAGS,
                train_manifest=DataConfig.from_flags(self.FLAGS.DATA.train).manifest,
                test_manifest=DataConfig.from_flags(self.FLAGS.DATA.test).manifest,
                legacy_source_root=getattr(self.FLAGS.SOLVER, "legacy_source_root", None),
            )

    def _save_iter_checkpoint(self, epoch: int, iteration: int) -> str:
        """Save a recovery checkpoint during long epochs.

        THSolver only checkpoints at epoch boundaries. Objaverse epochs are large
        enough that an interrupted job can otherwise lose hours of work.
        """
        if not self.is_master:
            return str(
                Path(self.logdir)
                / "checkpoints"
                / f"epoch{epoch:04d}_iter{iteration:08d}.solver.tar"
            )
        checkpoint_dir = Path(self.logdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dict = self._model_state_dict()
        checkpoint_name = checkpoint_dir / f"epoch{epoch:04d}_iter{iteration:08d}"
        model_path = Path(str(checkpoint_name) + ".model.pth")
        solver_path = Path(str(checkpoint_name) + ".solver.tar")
        metadata_path = Path(str(checkpoint_name) + ".json")
        self._atomic_torch_save(model_dict, model_path)
        self._atomic_torch_save(
            self._solver_checkpoint(model_dict, epoch=epoch, iteration=iteration), solver_path
        )
        self._atomic_json_save(
            {
                "epoch": int(epoch),
                "iteration": int(iteration),
                "world_size": int(self.world_size),
                "best_val": self.best_val,
            },
            metadata_path,
        )
        keep = max(int(getattr(self.FLAGS.SOLVER, "ckpt_num", 10)), 1)
        checkpoints = sorted(
            checkpoint_dir.glob("epoch*_iter*.solver.tar"), key=lambda path: path.stat().st_mtime
        )
        for old_solver in checkpoints[:-keep]:
            old_solver.unlink(missing_ok=True)
            model_path = Path(str(old_solver).removesuffix(".solver.tar") + ".model.pth")
            model_path.unlink(missing_ok=True)
            metadata_path = Path(str(old_solver).removesuffix(".solver.tar") + ".json")
            metadata_path.unlink(missing_ok=True)
        return str(solver_path)

    @torch.no_grad()
    def validate_checkpoint(self, epoch: int, iteration: int, checkpoint_path: str) -> None:
        """Evaluate immediately after an intra-epoch checkpoint.

        The resulting JSONL stream makes it possible to identify the first
        checkpoint where validation geometry starts to regress, instead of
        observing only epoch-boundary metrics.
        """
        was_training = self.model.training
        self.model.eval()
        self.test_iter = iter(self.test_loader)
        tracker = AverageTracker()
        limit = self.checkpoint_validation_batches
        if limit < 0:
            limit = len(self.test_loader)
        else:
            limit = min(limit, len(self.test_loader))
        for it in range(limit):
            batch = next(self.test_iter)
            batch["iter_num"] = it
            batch["epoch"] = epoch
            tracker.update(self.test_step(batch))
        if self.world_size > 1:
            tracker.average_all_gather()
        if self.is_master:
            values = tracker.average()
            record = {
                "epoch": int(epoch),
                "iteration": int(iteration),
                "checkpoint": str(checkpoint_path),
                "batches": int(limit),
                "metrics": values,
            }
            path = Path(self.logdir) / "checkpoint_validation.jsonl"
            with path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self.save_best_iter_checkpoint(tracker, epoch, iteration)
        if was_training:
            self.model.train()
            if bool(getattr(self.FLAGS.MODEL, "freeze_batch_norm", False)):
                self._freeze_batch_norm_statistics(self.model)

    def save_checkpoint(self, epoch: int) -> None:
        """Save immutable epoch milestones outside recovery-checkpoint rotation."""
        if not self.is_master:
            return
        checkpoint_dir = Path(self.logdir) / "checkpoints"
        checkpoint_name = checkpoint_dir / f"{epoch:05d}"
        model_dict = self._model_state_dict()
        self._atomic_torch_save(model_dict, Path(str(checkpoint_name) + ".model.pth"))
        self._atomic_torch_save(
            self._solver_checkpoint(model_dict, epoch=epoch),
            Path(str(checkpoint_name) + ".solver.tar"),
        )

    def save_best_checkpoint(self, tracker: AverageTracker, epoch: int) -> None:
        best_spec = str(self.FLAGS.SOLVER.best_val)
        if not best_spec or str(self.FLAGS.SOLVER.run) != "train":
            return
        compare, key = best_spec.split(":", 1)
        metric_key = "test/" + key
        if compare not in {"min", "max"} or metric_key not in tracker.value:
            return
        current = float((tracker.value[metric_key] / tracker.num[metric_key]).item())
        improved = self.best_val is None or (
            current < self.best_val if compare == "min" else current > self.best_val
        )
        if not improved:
            return
        self.best_val = current
        model_dict = self._model_state_dict()
        run_dir = Path(self.logdir)
        immutable = run_dir / f"best_epoch{epoch:04d}.model.pth"
        self._atomic_torch_save(model_dict, immutable)
        self._atomic_torch_save(model_dict, run_dir / "best_model.pth")
        message = f"epoch: {epoch}, {metric_key}: {current:.6f}"
        with (run_dir / "best_model.txt").open("a") as handle:
            handle.write(message + "\n")
        self._atomic_json_save(
            {
                "epoch": int(epoch),
                "metric": metric_key,
                "mode": compare,
                "value": current,
                "checkpoint": str(immutable),
            },
            run_dir / "best_model.json",
        )
        print("=> Best model at " + message)

    def save_best_iter_checkpoint(
        self, tracker: AverageTracker, epoch: int, iteration: int
    ) -> None:
        """Promote a validated intra-epoch state without losing its identity."""
        if not self.is_master:
            return
        best_spec = str(self.FLAGS.SOLVER.best_val)
        if not best_spec or str(self.FLAGS.SOLVER.run) != "train":
            return
        compare, key = best_spec.split(":", 1)
        metric_key = "test/" + key
        if compare not in {"min", "max"} or metric_key not in tracker.value:
            return
        current = float((tracker.value[metric_key] / tracker.num[metric_key]).item())
        improved = self._best_checkpoint_probe_val is None or (
            current < self._best_checkpoint_probe_val
            if compare == "min"
            else current > self._best_checkpoint_probe_val
        )
        if not improved:
            return
        self._best_checkpoint_probe_val = current
        model_dict = self._model_state_dict()
        run_dir = Path(self.logdir)
        immutable = run_dir / f"best_probe_epoch{epoch:04d}_iter{iteration:08d}.model.pth"
        self._atomic_torch_save(model_dict, immutable)
        self._atomic_torch_save(model_dict, run_dir / "best_probe_model.pth")
        message = (
            f"epoch: {epoch}, iteration: {iteration}, "
            f"{metric_key}: {current:.6f}"
        )
        with (run_dir / "best_model.txt").open("a") as handle:
            handle.write(message + "\n")
        self._atomic_json_save(
            {
                "epoch": int(epoch),
                "iteration": int(iteration),
                "metric": metric_key,
                "mode": compare,
                "value": current,
                "checkpoint": str(immutable),
                "validation_scope": "checkpoint_probe",
            },
            run_dir / "best_probe_model.json",
        )
        print("=> Best intra-epoch model at " + message, flush=True)

    def load_checkpoint(self):
        """Load THSolver checkpoints with exact intra-epoch resume support."""
        checkpoint = str(self.FLAGS.SOLVER.ckpt)
        if not checkpoint and os.path.isdir(getattr(self, "ckpt_dir", "")):
            candidates = sorted(
                name for name in os.listdir(self.ckpt_dir) if name.endswith("solver.tar")
            )
            if candidates:
                checkpoint = os.path.join(self.ckpt_dir, candidates[-1])

        self._resume_iteration = 0
        if not checkpoint:
            return
        trained = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if checkpoint.endswith(".solver.tar"):
            saved_world_size = trained.get("world_size")
            allow_mismatch = bool(
                getattr(self.FLAGS.SOLVER, "allow_world_size_mismatch_resume", False)
            )
            if (
                saved_world_size is not None
                and int(saved_world_size) != int(self.world_size)
                and "iteration" in trained
                and not allow_mismatch
            ):
                raise ValueError(
                    "Refusing intra-epoch solver resume with a different world size: "
                    f"checkpoint={saved_world_size}, current={self.world_size}. "
                    "Resume from an epoch milestone or explicitly set "
                    "SOLVER.allow_world_size_mismatch_resume true."
                )
            model_dict = trained["model_dict"]
            self.start_epoch = int(trained["epoch"]) + 1
            self.optimizer.load_state_dict(trained["optimizer_dict"])
            self.scheduler.load_state_dict(trained["scheduler_dict"])
            if self.use_amp and "scaler_dict" in trained:
                self.scaler.load_state_dict(trained["scaler_dict"])
            self.best_val = trained.get("best_val")
        else:
            model_dict = trained
        model = self.model.module if self.world_size > 1 else self.model
        model.load_state_dict(model_dict)
        if self.is_master:
            print(f"Load the checkpoint: {checkpoint}")
            print(f"The start_epoch is {self.start_epoch}")
        match = re.search(r"epoch(\d+)_iter(\d+)\.solver\.tar$", checkpoint)
        if match is None:
            return
        self.start_epoch = int(match.group(1))
        self._resume_iteration = int(match.group(2))
        if self.is_master:
            print(
                f"Resume intra-epoch checkpoint at epoch {self.start_epoch}, "
                f"iteration {self._resume_iteration}"
            )

    def _preprocessor(self, flags) -> DeviceBatchPreprocessor:
        key = id(flags)
        if key not in self._preprocessors:
            self._preprocessors[key] = DeviceBatchPreprocessor(
                DataConfig.from_flags(flags), torch.device("cuda", self.device)
            )
        return self._preprocessors[key]

    def process_batch(self, batch: dict[str, object], flags) -> PreparedBatch:
        samples = batch.get("samples")
        if not isinstance(samples, list):
            raise TypeError("DataLoader batch must contain a list under 'samples'")
        # The device preprocessor builds OCNN sparse structures, camera rays and
        # supervision coordinates.  It is called from inside the solver's
        # outer AMP context, but these geometric quantities should never be
        # autocast to FP16.  In particular, depth/ray normalization and octree
        # coordinates are later used as inputs to the numerically sensitive
        # geometry head.
        with torch.autocast(device_type="cuda", enabled=False):
            return self._preprocessor(flags)(samples)

    def _model_forward(self, prepared: PreparedBatch):
        input_features = build_input_features(prepared.octree, self.FLAGS.MODEL.feature)
        if input_features.shape[1] != self.FLAGS.MODEL.channel:
            raise RuntimeError(
                f"Built {input_features.shape[1]} input channels, expected {self.FLAGS.MODEL.channel}"
            )
        prediction = self.model(
            input_features,
            prepared.octree,
            prepared.octree.depth,
            prepared.pins,
            prepared.directions,
            prepared.voxel_indices,
            prepared.pouts,
            # 0 disables chunking; a positive value bounds the renderer activation
            # peak by splitting the candidate dimension (see NeuralIntersectionModel).
            chunk_size=int(getattr(self.FLAGS.MODEL, "forward_chunk_size", 0) or 0),
        )
        target = IntersectionTarget(
            prepared.local_positions,
            prepared.normals,
            prepared.colors,
            prepared.hits,
            prepared.pins,
            prepared.pouts,
        )
        return prediction, target

    @staticmethod
    def _metrics(result, *, detailed: bool = False) -> dict[str, torch.Tensor]:
        metrics = {
            "psnr_c": image_psnr(result.predicted_color, result.target_color),
            "psnr_n": image_psnr(result.predicted_normal, result.target_normal),
            "masked_psnr_c": masked_psnr(
                result.predicted_color, result.target_color, result.target_hit_mask
            ),
            "masked_psnr_n": masked_psnr(
                result.predicted_normal, result.target_normal, result.target_hit_mask
            ),
        }
        metrics.update(
            hit_classification_metrics(result.predicted_hit_mask, result.target_hit_mask)
        )
        metrics.update(hit_topology_metrics(result.predicted_hit_mask, result.target_hit_mask))
        if not detailed:
            return metrics
        noise_rows = []
        detail_rows = []
        color_rows = []
        for batch_id in range(result.target_hit_mask.shape[0]):
            noise_rows.append(
                geometry_noise_metrics(
                    result.predicted_normal[batch_id],
                    result.target_normal[batch_id],
                    result.predicted_hit_mask[batch_id],
                    result.target_hit_mask[batch_id],
                )
            )
            detail_rows.append(
                normal_detail_metrics(
                    result.predicted_normal[batch_id],
                    result.target_normal[batch_id],
                    result.predicted_hit_mask[batch_id],
                    result.target_hit_mask[batch_id],
                )
            )
            color_rows.append(
                color_detail_metrics(
                    result.predicted_color[batch_id],
                    result.target_color[batch_id],
                    result.target_hit_mask[batch_id],
                )
            )
        for name in noise_rows[0]:
            metrics[name] = torch.stack([row[name] for row in noise_rows]).mean()
        for name in detail_rows[0]:
            metrics[name] = torch.stack([row[name] for row in detail_rows]).mean()
        for name in color_rows[0]:
            metrics[name] = torch.stack([row[name] for row in color_rows]).mean()
        return metrics

    @staticmethod
    def _mem_trace(label: str) -> None:
        """Temporary diagnostic: peak/current device memory at a labelled point."""
        import os

        if not os.environ.get("NEUOCT_MEM_TRACE"):
            return
        if not torch.cuda.is_available():
            return
        print(
            f"[mem] {label:24s} cur {torch.cuda.memory_allocated() / 2**30:6.2f} GiB  "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB",
            flush=True,
        )

    def train_step(self, batch):
        prepared = self.process_batch(batch, self.FLAGS.DATA.train)
        self._mem_trace(f"after prepare (pins={prepared.pins.shape[0]})")
        prediction, target = self._model_forward(prepared)
        self._mem_trace("after forward")
        sample_weights = normal_detail_weights_for_pairs(
            prepared.normals,
            prepared.hits,
            prepared.ray_labels,
            batch_size=prepared.batch_size,
            height=prepared.image_height,
            width=prepared.image_width,
            scale=self.detail_weight_scale,
        )
        color_sample_weights = color_detail_weights_for_pairs(
            prepared.colors,
            prepared.hits,
            prepared.ray_labels,
            batch_size=prepared.batch_size,
            height=prepared.image_height,
            width=prepared.image_width,
            detail_scale=self.color_detail_weight_scale,
            rarity_scale=self.color_rarity_weight_scale,
        )
        self._mem_trace("after weights")
        result = self.objective(
            prediction,
            target,
            prepared.ray_labels,
            batch_size=prepared.batch_size,
            height=prepared.image_height,
            width=prepared.image_width,
            epoch=int(batch.get("epoch", 0)),
            total_epochs=int(self.FLAGS.SOLVER.max_epoch),
            training=True,
            sample_weights=sample_weights,
            color_sample_weights=color_sample_weights,
        )
        output = {"train/loss": result.loss}
        output.update({f"train/{key}": value for key, value in self._metrics(result).items()})
        output.update({f"train/loss_{key}": value for key, value in result.components.items()})
        parameter_l2 = torch.zeros_like(result.loss)
        parameter_l1 = torch.zeros_like(result.loss)
        if self.parameter_l2 > 0 or self.parameter_l1 > 0:
            params = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if params:
                parameter_l2 = torch.stack(
                    [parameter.float().square().mean() for parameter in params]
                ).mean()
                parameter_l1 = torch.stack(
                    [parameter.float().abs().mean() for parameter in params]
                ).mean()
                output["train/loss"] = (
                    result.loss
                    + self.parameter_l2 * parameter_l2
                    + self.parameter_l1 * parameter_l1
                )
        output["train/loss_parameter_l2"] = parameter_l2
        output["train/loss_parameter_l1"] = parameter_l1
        self._last_train_prediction = prediction
        self._last_train_entries = prepared.entries
        return output

    def _loss_is_finite_on_all_ranks(self, loss: torch.Tensor) -> tuple[bool, bool]:
        local_finite = bool(torch.isfinite(loss.detach()).all())
        global_flag = torch.tensor(
            int(local_finite), device=loss.device, dtype=torch.int32
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_flag, op=torch.distributed.ReduceOp.MIN)
        return local_finite, bool(global_flag.item())

    def _gradients_are_finite_on_all_ranks(self) -> tuple[bool, bool]:
        """Check gradients before clipping/stepping, synchronized under DDP."""
        local_finite = True
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                local_finite = False
                break
        global_flag = torch.tensor(
            int(local_finite), device=f"cuda:{self.device}", dtype=torch.int32
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_flag, op=torch.distributed.ReduceOp.MIN)
        return local_finite, bool(global_flag.item())

    def _log_nonfinite_train_step(
        self, epoch: int, iteration: int, output: dict[str, torch.Tensor]
    ) -> None:
        entries = getattr(self, "_last_train_entries", ())
        entry_names = [entry.relative_name for entry in entries]
        bad_outputs = [
            key
            for key, value in output.items()
            if torch.is_tensor(value) and not bool(torch.isfinite(value.detach()).all())
        ]
        bad_stages = []
        prediction = getattr(self, "_last_train_prediction", None)
        if prediction is not None:
            for stage_index, stage in enumerate(prediction.stages):
                for field_name in ("position", "normal", "color", "hit_probability"):
                    value = getattr(stage, field_name).detach()
                    invalid = int((~torch.isfinite(value)).sum().item())
                    if invalid:
                        bad_stages.append(f"stage{stage_index}.{field_name}={invalid}")
        rank = int(getattr(self, "rank", 0))
        print(
            "Non-finite training loss; skipping synchronized optimizer step: "
            f"rank={rank}, epoch={epoch}, iter={iteration}, entries={entry_names}, "
            f"outputs={bad_outputs}, stages={bad_stages}",
            flush=True,
        )

    def train_epoch(self, epoch):
        checkpoint_every = int(getattr(self.FLAGS.SOLVER, "checkpoint_every_iter", 0))
        if checkpoint_every <= 0:
            return super().train_epoch(epoch)

        self.model.train()
        if bool(getattr(self.FLAGS.MODEL, "freeze_batch_norm", False)):
            frozen_count = self._freeze_batch_norm_statistics(self.model)
            if self.is_master and epoch == self.start_epoch:
                print(f"Frozen BatchNorm running statistics in {frozen_count} modules")
        flags = self.FLAGS.SOLVER
        sampler = self.train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        start_iteration = self._resume_iteration if epoch == self.start_epoch else 0
        if start_iteration < 0 or start_iteration > len(self.train_loader):
            raise ValueError(
                f"Resume iteration {start_iteration} is outside epoch length "
                f"{len(self.train_loader)}"
            )
        if start_iteration:
            sample_offset = start_iteration * int(self.FLAGS.DATA.train.batch_size)
            if hasattr(sampler, "seek"):
                sampler.seek(sample_offset)
            else:
                for _ in range(sample_offset):
                    next(sampler)
        # Rebuild the iterator after setting the epoch/offset so prefetched indices
        # from the iterator constructed by THSolver cannot leak across a resume.
        self.train_iter = iter(self.train_loader)
        avg_tracker = AverageTracker()
        skipped_nonfinite = 0
        rng = range(start_iteration, len(self.train_loader))
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            if flags.empty_cache > 0 and it % flags.empty_cache == 0:
                torch.cuda.empty_cache()
            batch = next(self.train_iter)
            batch["iter_num"] = it
            batch["epoch"] = epoch
            self.optimizer.zero_grad(flags.zero_grad_to_none)
            with torch.autocast("cuda", enabled=self.use_amp):
                output = self.train_step(batch)
                loss = output["train/loss"]
            local_finite, all_finite = self._loss_is_finite_on_all_ranks(loss)
            if not all_finite:
                skipped_nonfinite += 1
                if not local_finite:
                    self._log_nonfinite_train_step(epoch, it, output)
                if self.world_size > 1:
                    # Complete DDP's reduction hooks on every rank, then discard
                    # the gradients without changing optimizer or scaler state.
                    torch.nan_to_num(
                        loss.float(), nan=0.0, posinf=0.0, neginf=0.0
                    ).backward()
                self.optimizer.zero_grad(flags.zero_grad_to_none)
                self._last_train_prediction = None
                self._last_train_entries = ()
                continue
            clip_grad = flags.clip_grad
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Unscale before inspecting or clipping, otherwise a finite
                # FP32 gradient can be mistaken for overflow because it still
                # contains the GradScaler scale factor.
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            local_grad_finite, all_grad_finite = self._gradients_are_finite_on_all_ranks()
            if not all_grad_finite:
                skipped_nonfinite += 1
                if not local_grad_finite:
                    print(
                        "Non-finite gradient; skipping synchronized optimizer step: "
                        f"rank={getattr(self, 'rank', 0)}, epoch={epoch}, iter={it}",
                        flush=True,
                    )
                self.optimizer.zero_grad(flags.zero_grad_to_none)
                # unscale_() has recorded the overflow for AMP, allowing the
                # scaler to reduce its scale even though no step is taken.
                if self.use_amp:
                    self.scaler.update()
                continue

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            avg_tracker.update(output)
            self._last_train_prediction = None
            self._last_train_entries = ()
            avg_tracker.record_time()
            if self.is_master and flags.log_per_iter > 0 and it % flags.log_per_iter == 0:
                avg_tracker.log(epoch, msg_tag="- ", notes=f"iter: {it}", print_time=False)
            if (it + 1) % checkpoint_every == 0:
                checkpoint_path = self._save_iter_checkpoint(epoch, it + 1)
                if self.world_size > 1:
                    torch.distributed.barrier()
                if self.validate_each_checkpoint:
                    self.validate_checkpoint(epoch, it + 1, checkpoint_path)
        self._resume_iteration = 0
        if self.world_size > 1:
            avg_tracker.average_all_gather()
        if self.is_master:
            print(f"Skipped non-finite training steps: {skipped_nonfinite}")
            avg_tracker.log(epoch, self.summary_writer, print_time=True)

    @torch.no_grad()
    def test_step(self, batch):
        prepared = self.process_batch(batch, self.FLAGS.DATA.test)
        prediction, target = self._model_forward(prepared)
        self._mem_trace("after weights")
        result = self.objective(
            prediction,
            target,
            prepared.ray_labels,
            batch_size=prepared.batch_size,
            height=prepared.image_height,
            width=prepared.image_width,
            training=False,
        )
        metrics = self._metrics(result, detailed=True)
        selection_score = (
            metrics["hit_iou"]
            + 0.25 * metrics["hit_boundary_f1"]
            - metrics["interior_hole_rate"]
            - 0.5 * metrics["remote_floater_rate"]
            + 0.01 * metrics["masked_psnr_c"]
            - 0.005 * metrics["masked_normal_angle_error_deg"]
        )
        output = {"test/loss": result.loss, "test/selection_score": selection_score}
        output.update({f"test/{key}": value for key, value in metrics.items()})
        output.update({f"test/loss_{key}": value for key, value in result.components.items()})
        return output

    def _inference_scene(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        batch = next(self.test_iter)
        prepared = self.process_batch(batch, self.FLAGS.DATA.test)
        input_features = build_input_features(prepared.octree, self.FLAGS.MODEL.feature)
        model = self.model.module if self.world_size > 1 else self.model
        with torch.no_grad():
            voxel_features, field = model.extract_scene_features(
                input_features, prepared.octree, prepared.octree.depth
            )
        return batch["samples"], prepared, model, voxel_features, field

    @torch.no_grad()
    def bn_calibrate(self) -> None:
        """Re-estimate BatchNorm buffers from representative training objects."""
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader()
        self.load_checkpoint()
        model = self.model.module if self.world_size > 1 else self.model
        batch_norms = [module for module in model.modules() if isinstance(module, _BatchNorm)]
        if not batch_norms:
            raise RuntimeError("The model has no BatchNorm modules to calibrate")
        reset = bool(getattr(self.FLAGS.SOLVER, "bn_calibration_reset", True))
        original_momenta = [module.momentum for module in batch_norms]
        for module in batch_norms:
            if reset:
                module.reset_running_stats()
            module.train()
            module.momentum = None
        model.train()
        steps = int(getattr(self.FLAGS.SOLVER, "bn_calibration_steps", 1024))
        steps = min(max(steps, 1), len(self.train_loader))
        sampler = self.train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        self.train_iter = iter(self.train_loader)
        for _ in tqdm(range(steps), ncols=80, disable=self.disable_tqdm):
            batch = next(self.train_iter)
            prepared = self.process_batch(batch, self.FLAGS.DATA.train)
            with torch.autocast("cuda", enabled=self.use_amp):
                self._model_forward(prepared)
        for module, momentum in zip(batch_norms, original_momenta, strict=True):
            module.momentum = momentum
        output_text = str(getattr(self.FLAGS.SOLVER, "bn_calibration_output", "")).strip()
        if not output_text:
            raise ValueError("SOLVER.bn_calibration_output is required")
        output = Path(output_text)
        if self.is_master:
            self._atomic_torch_save(model.state_dict(), output)
            self._atomic_json_save(
                {
                    "source_checkpoint": str(self.FLAGS.SOLVER.ckpt),
                    "steps": steps,
                    "reset_running_stats": reset,
                    "batch_norm_modules": len(batch_norms),
                    "output": str(output),
                },
                output.with_suffix(output.suffix + ".json"),
            )
            print(f"Saved BN-calibrated checkpoint: {output}")

    def render(self):
        samples, prepared, model, _, field = self._inference_scene()
        render_flags = getattr(self.FLAGS, "RENDER", None)
        max_batch_size = int(getattr(render_flags, "max_batch_size", 262144))
        hit_threshold = float(getattr(render_flags, "hit_threshold", 0.5))
        background = tuple(float(x) for x in getattr(render_flags, "background_color", (1, 1, 1)))
        output_root = (
            Path(str(getattr(self.FLAGS.SOLVER, "output_dir", Path(self.logdir) / "outputs")))
            / "render"
        )
        render_data = DataConfig.from_flags(self.FLAGS.DATA.test)
        for batch_id, sample in enumerate(samples):
            renderer = SparseRenderer(
                model.renderer_model,
                field,
                prepared.octree,
                depth=self.FLAGS.MODEL.q_depth,
                point_scale=render_data.point_scale,
                ray_scale=render_data.ray_scale,
                camera_convention=render_data.camera_convention,
                max_batch_size=max_batch_size,
                hit_threshold=hit_threshold,
                background_color=background,
            )
            object_dir = output_root / sample.entry.category / sample.entry.object_id
            output = renderer.render_frame(
                sample.camera_to_world, sample.intrinsics, batch_id=batch_id
            )
            save_render_output(output, object_dir, f"{sample.entry.frame_id:03d}")
            if bool(getattr(self.FLAGS.SOLVER, "save_video", False)):
                view_count = int(getattr(render_flags, "video_views", 120))
                views = generate_orbit_views(
                    view_count,
                    camera_convention=render_data.camera_convention,
                    up_axis=render_data.orbit_up_axis,
                    device=field.features.device,
                )
                for view_id, transform in enumerate(views):
                    frame = renderer.render_frame(transform, sample.intrinsics, batch_id=batch_id)
                    save_render_output(
                        frame, object_dir / "orbit", f"{view_id:03d}", save_composite=True
                    )

    def export_features(self):
        _, prepared, _, voxel_features, field = self._inference_scene()
        x, y, z, batch = prepared.octree.xyzb(self.FLAGS.MODEL.q_depth, nempty=True)
        coordinates = torch.stack([batch, x, y, z], dim=1)
        output_root = (
            Path(str(getattr(self.FLAGS.SOLVER, "output_dir", Path(self.logdir) / "outputs")))
            / "features"
        )
        names = [entry.relative_name for entry in prepared.entries]
        filename = "__".join(name.replace("/", "-") for name in names) + ".pt"
        path = save_feature_export(
            output_root / filename,
            entries=names,
            query_depth=self.FLAGS.MODEL.q_depth,
            voxel_coordinates=coordinates,
            voxel_features=voxel_features,
            field=field,
        )
        if self.is_master:
            print(f"Saved feature export: {path}")

    def gen_sp(self):
        self.export_features()

    def compare_reconstruction(self):
        samples, prepared, model, _, original_field = self._inference_scene()
        render_flags = getattr(self.FLAGS, "RENDER", None)
        reconstruction_value = str(getattr(render_flags, "reconstruction_path", ""))
        reconstruction_path = Path(reconstruction_value)
        if not reconstruction_value.strip() or not reconstruction_path.is_file():
            raise FileNotFoundError(
                "RENDER.reconstruction_path must point to a schema-v1 feature export"
            )
        payload = load_feature_export(reconstruction_path)
        expected_entries = [entry.relative_name for entry in prepared.entries]
        if payload["entries"] != expected_entries:
            raise ValueError(
                f"Reconstruction entries {payload['entries']} do not match batch {expected_entries}"
            )
        if int(payload["query_depth"]) != int(self.FLAGS.MODEL.q_depth):
            raise ValueError("Reconstruction query depth does not match MODEL.q_depth")
        x, y, z, batch = prepared.octree.xyzb(self.FLAGS.MODEL.q_depth, nempty=True)
        current_coordinates = torch.stack([batch, x, y, z], dim=1).cpu().to(torch.int32)
        if not torch.equal(current_coordinates, payload["voxel_coordinates"]):
            raise ValueError(
                "Schema-v1 reconstruction must use exactly the current octree voxel ordering"
            )
        device = original_field.features.device
        reconstructed_field = VertexFeatureField(
            VertexIndex(
                payload["vertex_coordinates"].to(device),
                payload["voxel_vertex_indices"].to(device),
            ),
            payload["vertex_features"].to(device),
        )
        output_root = (
            Path(str(getattr(self.FLAGS.SOLVER, "output_dir", Path(self.logdir) / "outputs")))
            / "reconstruction_compare"
        )
        for batch_id, sample in enumerate(samples):
            render_data = DataConfig.from_flags(self.FLAGS.DATA.test)
            for name, field in (
                ("original", original_field),
                ("reconstructed", reconstructed_field),
            ):
                renderer = SparseRenderer(
                    model.renderer_model,
                    field,
                    prepared.octree,
                    depth=self.FLAGS.MODEL.q_depth,
                    point_scale=render_data.point_scale,
                    ray_scale=render_data.ray_scale,
                    camera_convention=render_data.camera_convention,
                )
                output = renderer.render_frame(
                    sample.camera_to_world, sample.intrinsics, batch_id=batch_id
                )
                save_render_output(
                    output,
                    output_root / sample.entry.category / sample.entry.object_id / name,
                    f"{sample.entry.frame_id:03d}",
                )

    @classmethod
    def worker(cls, rank, flags):
        world_size = cls.get_world_size(flags)
        if flags.SOLVER.ddp_mode == "spawn":
            gpu_rank = flags.SOLVER.gpu[rank]
            torch.cuda.set_device(gpu_rank)
            if world_size > 1:
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method=f"tcp://localhost:{flags.SOLVER.port}",
                    world_size=world_size,
                    rank=rank,
                )
                torch.distributed.barrier(device_ids=[gpu_rank])
        elif flags.SOLVER.ddp_mode == "torchrun":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            if world_size > 1:
                torch.distributed.init_process_group(backend="nccl", init_method="env://")
                torch.distributed.barrier(device_ids=[local_rank])
        else:
            raise NotImplementedError(f"Unsupported DDP mode: {flags.SOLVER.ddp_mode}")
        the_solver = cls(flags, rank == 0)
        the_solver.rank = rank
        the_solver.run()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
