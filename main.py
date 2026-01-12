import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure

import os
import math
from tqdm import tqdm
from time import time
from typing import Tuple, List

from modules import NeuralIntersectionModel5 as NeuralIntersectionModel
from modules import PNHLoss as IntersectionLoss
# from models import IdenticalLocalPositionEncoder as LocalPositionEncoder
# from models import ZeroMeanLocalPositionEncoder as LocalPositionEncoder
# from models import SymmetricLocalPositionEncoder as LocalPositionEncoder
from modules import LocalPositionEncoder as LocalPositionEncoder
from modules import VoxelFeatureModel as VoxelFeatureModel
from modules import VoxelRendererModel5 as VoxelRenderModel
# from modules import RenderingLoss4 as RenderingLoss
from RayDataset import TrainingRayDataset, Renderer

from Arguments import args, config_name, object_name, f_dim, num_epochs, rendering_batch_size, port, depth, use_tqdm, n_iters # , loss_weights
from sparse_voxels import SparseVoxels

torch.set_float32_matmul_precision('high')
local_position_encode = LocalPositionEncoder(depth)

def get_gaussian_loss_weights(epoch, total_eopchs = num_epochs, num_groups = n_iters, sigma = 0.2, steepness = n_iters):
    progress = (epoch / total_eopchs - 0.5) * 2 + 0.5
    def sigmoid(x):
        return 1 / (1 + math.exp(-steepness * (x - 0.5)))
    mu = (sigmoid(progress) - 0.5) / (2 * sigmoid(1) - 1) + 0.5
    x = torch.linspace(0, 1, num_groups)
    group_weights = torch.exp(-torch.square((x - mu) / sigma) / 2)
    group_weights_normalized = group_weights / torch.sum(group_weights)
    return group_weights_normalized

def identity(batch):
    return batch[0]

def ddp_setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 训练流程框架
def train_ddp(rank, world_size, dataset: TrainingRayDataset, num_epochs, sparse_voxels: SparseVoxels):
    ddp_setup(rank, world_size)

    # 数据加载
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=identity,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # 初始化模型
    model = NeuralIntersectionModel(VoxelFeatureModel(sparse_voxels, f_dim), VoxelRenderModel(f_dim)).to(rank)
    if rank == 0:
        for layer_name, layer_weights in model.state_dict().items():
            print(f'{layer_name:60s}: {f"{layer_weights.shape}":40s} | {layer_weights.dtype}')
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    i_loss_fn = IntersectionLoss()
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(rank)
    # r_loss_fn = RenderingLoss().to(rank)
    
    if rank == 0:
        writer = SummaryWriter(log_dir=f'./logs/{config_name}/{object_name}')
        if not use_tqdm:
            start_time = time()

    # 训练循环
    for epoch in range(1, 1+num_epochs):
        
        sampler.set_epoch(epoch)
        ddp_model.train()
        total_iloss = 0
        # total_rloss = 0
        total_samples = 0
        loss_weights = get_gaussian_loss_weights(epoch)

        if rank == 0 and use_tqdm:
            pbar = tqdm(total=len(dataloader), desc = f'[Epoch {epoch:4d} / {num_epochs:4d}] iLoss = ?', unit='batch', dynamic_ncols=True)

        try:
            for batch in dataloader:
                pin, pout, indices, tgt_hit, tgt_pos, tgt_norm, labels, grid_base = [x.to(rank) for x in batch]
                # tgt_position_map = dataset.remapping_image(labels, tgt_pos, fill=0)
                tgt_normal_map = dataset.remapping_image(labels, tgt_norm, fill=0)
                
                B = pin.shape[0]
                tgt_local_pos = local_position_encode.encode(tgt_pos, tgt_hit > 0.5)
                
                # if False:
                #     pin_lambda = torch.clip(torch.rand((B, 1), device=rank) * 2, 0, 1)
                #     pos_lambda = 1 - pin_lambda
                #     position = pin * pin_lambda + tgt_pos * pos_lambda
                #     directions = F.normalize(pout - pin, dim=1)
                # else:
                position = pin
                directions = F.normalize(pout - pin, dim=1)
                tgt_intersection = (tgt_local_pos, tgt_norm, tgt_hit)

                # clamp pred_hit to [0, 1]
                # pred_hit = torch.clamp(pred_hit, 0.0, 1.0)
                
                optimizer.zero_grad()
                intersections = ddp_model(position, directions, indices, grid_base)
                loss = 0.0

                for loss_weight, intersection in list(zip(loss_weights, intersections)):
                    pred_pos, pred_norm, pred_hit = intersection

                    pred_hit_mask = pred_hit > 0.5
                    if pred_hit_mask.any():
                        # pred_hit_pos = pred_pos[pred_hit_mask]
                        pred_hit_norm = pred_norm[pred_hit_mask]
                        pred_hit_rays = labels[pred_hit_mask]
                        unique_ray_mask = torch.cat([torch.tensor([1], dtype=torch.bool, device=labels.device), pred_hit_rays[1:] != pred_hit_rays[:-1]], dim=0)
                        # unique_pred_hit_pos = pred_hit_pos[unique_ray_mask]
                        unique_pred_hit_norm = pred_hit_norm[unique_ray_mask]
                        unique_pred_hit_rays = pred_hit_rays[unique_ray_mask]
                        # pred_position_map = dataset.remapping_image(unique_pred_hit_rays, unique_pred_hit_pos, fill=0)
                        pred_normal_map = dataset.remapping_image(unique_pred_hit_rays, unique_pred_hit_norm, fill=0)
                    else:
                        # pred_position_map = torch.zeros_like(tgt_position_map)
                        pred_normal_map = torch.full_like(tgt_normal_map, fill_value=0)
                    # ssim_pos = 1.0 - ssim_loss(pred_position_map.permute(2,0,1).unsqueeze(0), tgt_position_map.permute(2,0,1).unsqueeze(0))
                    ssim_norm = 1.0 - ssim_loss(pred_normal_map.permute(2,0,1).unsqueeze(0), tgt_normal_map.permute(2,0,1).unsqueeze(0))

                    iloss = i_loss_fn(intersection, tgt_intersection)
                    loss = (iloss + ssim_norm * 0.5) * loss_weight + loss

                # check if loss is nan or inf
                # if not torch.isfinite(loss):
                #     print(f'Rank {rank} | Epoch {epoch} | Invalid loss: {loss.item()}', flush=True)
                #     continue

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

                optimizer.step()

                total_iloss += iloss.item()
                total_samples += 1

                if rank == 0 and use_tqdm:
                    pbar.set_description(f'[Epoch {epoch:4d} / {num_epochs:4d}] iLoss = {iloss.item():12.8f}')
                    pbar.update(1)
    
        except KeyboardInterrupt:
            if rank == 0:
                if use_tqdm:
                    pbar.close()
                print('Training interrupted by user', flush=True)
            raise KeyboardInterrupt
        
        iloss_tensor = torch.tensor(total_iloss).to(rank)
        sample_tensor = torch.tensor(total_samples).to(rank)

        dist.all_reduce(iloss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)

        global_avg_iloss = iloss_tensor.item() / sample_tensor.item()
        scheduler.step()

        if rank == 0:
            writer.add_scalar('iLoss', global_avg_iloss, epoch)
            if use_tqdm:
                pbar.set_description(f'[Epoch {epoch:4d} / {num_epochs:4d}] Avg iLoss = {global_avg_iloss:12.8f}')
                pbar.close()
            else:
                end_time = time()
                delta_time = end_time - start_time
                start_time = time()
                print(f'{delta_time:5.2f}s | [Epoch {epoch:4d} / {num_epochs:4d}] Avg iLoss = {global_avg_iloss:12.8f}', flush=True)

            if epoch % 100 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'feature': ddp_model.module.feature_model.state_dict(),
                    'renderer': ddp_model.module.renderer_model.state_dict(),
                }
                torch.save(checkpoint, f"./checkpoint/{config_name}/{object_name}/models_{epoch}.pth")
    
    if rank == 0:
        writer.close()
        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'feature': ddp_model.module.feature_model.cpu().state_dict(),
            'renderer': ddp_model.module.renderer_model.cpu().state_dict(),
        }
        torch.save(checkpoint, f"./checkpoint/{config_name}/{object_name}/models.pth")

    # 清理分布式环境
    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()

def train(dataset: TrainingRayDataset, num_epochs, sparse_voxels: SparseVoxels, device=torch.device('cuda')):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=identity,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    # 初始化模型
    model = NeuralIntersectionModel(VoxelFeatureModel(sparse_voxels, f_dim), VoxelRenderModel(f_dim)).to(device)
    # model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    i_loss_fn = IntersectionLoss()
    # r_loss_fn = RenderingLoss().to(device)
    
    if not use_tqdm:
        start_time = time()

    # 训练循环
    for epoch in range(1, 1+num_epochs):
        
        model.train()
        total_iloss = 0
        # total_rloss = 0
        total_samples = 0

        if use_tqdm:
            pbar = tqdm(total=len(dataloader), desc = f'[Epoch {epoch:4d} / {num_epochs:4d}] iLoss = ? | rLoss = ?', unit='image', dynamic_ncols=True)

        try:
            for batch in dataloader:
                pin, pout, indices, tgt_hit, tgt_pos, tgt_norm, tgt_col, labels = [x.to(device) for x in batch]
                # print(pin.shape)
                # tgt_position_map = dataset.remapping_image(labels, tgt_pos, fill=0)
                # tgt_normal_map = dataset.remapping_image(labels, tgt_norm, fill=0.57735)
                
                optimizer.zero_grad()
                pred_pos, pred_norm, pred_col, pred_hit = model(pin, pout, indices)

                # pred_hit_mask = pred_hit > 0.5
                # if pred_hit_mask.any():
                #     pred_hit_pos = pred_pos[pred_hit_mask]
                #     pred_hit_norm = pred_norm[pred_hit_mask]
                #     pred_hit_rays = labels[pred_hit_mask]
                #     unique_ray_mask = torch.cat([torch.tensor([1], dtype=torch.bool, device=labels.device), pred_hit_rays[1:] != pred_hit_rays[:-1]], dim=0)
                #     unique_pred_hit_pos = pred_hit_pos[unique_ray_mask]
                #     unique_pred_hit_norm = pred_hit_norm[unique_ray_mask]
                #     unique_pred_hit_rays = pred_hit_rays[unique_ray_mask]
                #     pred_position_map = dataset.remapping_image(unique_pred_hit_rays, unique_pred_hit_pos, fill=0)
                #     pred_normal_map = dataset.remapping_image(unique_pred_hit_rays, unique_pred_hit_norm, fill=0.57735)
                # else:
                #     pred_position_map = torch.zeros_like(tgt_position_map)
                #     pred_normal_map = torch.full_like(tgt_normal_map, 0.57735)
                
                iloss = i_loss_fn((pred_pos, pred_norm, pred_col, pred_hit), (tgt_pos, tgt_norm, tgt_col, tgt_hit))
                # rloss = r_loss_fn(pred_position_map, pred_normal_map, tgt_position_map, tgt_normal_map)
                loss = iloss

                loss.backward()
                optimizer.step()

                total_iloss += iloss.item()
                # total_rloss += rloss.item()
                total_samples += 1

                if use_tqdm:
                    pbar.set_description(f'[Epoch {epoch:4d} / {num_epochs:4d}] iLoss = {iloss.item():12.8f}')
                    pbar.update(1)
    
        except KeyboardInterrupt:
            if use_tqdm:
                pbar.close()
            print('Training interrupted by user', flush=True)
            raise KeyboardInterrupt
        
        global_avg_iloss = total_iloss / total_samples
        # global_avg_rloss = total_rloss / total_samples
        
        if use_tqdm:
            pbar.set_description(f'[Epoch {epoch:4d} / {num_epochs:4d}] Avg iLoss = {global_avg_iloss:12.8f}')
            pbar.close()
        else:
            end_time = time()
            delta_time = end_time - start_time
            start_time = time()
            print(f'{delta_time:5.2f}s | [Epoch {epoch:4d} / {num_epochs:4d}] Avg iLoss = {global_avg_iloss:12.8f}', flush=True)

        if epoch % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'feature': model.feature_model.state_dict(),
                'renderer': model.renderer_model.state_dict(),
            }
            torch.save(checkpoint, f"./checkpoint/{config_name}/{object_name}/models_{epoch}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'feature': model.feature_model.cpu().state_dict(),
        'renderer': model.renderer_model.cpu().state_dict(),
    }
    torch.save(checkpoint, f"./checkpoint/{config_name}/{object_name}/models.pth")


def main():
    print(f'arguments: {args}', flush=True)

    print(f'pid: {os.getpid()}', flush=True)
    world_size = torch.cuda.device_count()
    print(f'world size = {world_size}', flush=True)

    data_path = f'./data/{object_name}'
    # dataset = TrainingRayDataset(data_path, depth=depth, device='cuda', cache_first=False)
    # raise NotImplementedError
    dataset = TrainingRayDataset(data_path, depth=depth, device='cpu')
    sparse_voxels = dataset.sparse_voxels

    os.makedirs(f'./checkpoint/{config_name}/{object_name}', exist_ok=True)

    if world_size > 1:
        training_args = (world_size, dataset, num_epochs, sparse_voxels)
        mp.spawn(train_ddp, args=training_args, nprocs=world_size, join=True)
    else:
        train(dataset, num_epochs, sparse_voxels, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    chackpoint_path = f'./checkpoint/{config_name}/{object_name}/models.pth'
    checkpoint = torch.load(chackpoint_path, weights_only=False)

    feature_model = VoxelFeatureModel(f_dim=f_dim)
    renderer_model = VoxelRenderModel(f_dim)
    feature_model.load_state_dict(checkpoint['feature'])
    renderer_model.load_state_dict(checkpoint['renderer'])
    model = NeuralIntersectionModel(feature_model, renderer_model)
    model.to(device)
    model.eval()

    training_views = dataset.transform_matrices.to(device)
    training_renderer = Renderer(model, sparse_voxels, training_views, depth=depth, max_batch_size=rendering_batch_size, device=device)
    training_renderer.render(save_images=False, save_video=False, output_dir=f'./training_views/{config_name}', object_name=object_name)
    training_renderer.render(save_images=True, save_video=False, output_dir=f'./training_views/{config_name}', object_name=object_name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    new_views = Renderer.generate_views(240, device=device)
    renderer = Renderer(model, sparse_voxels, new_views, depth=depth, max_batch_size=rendering_batch_size, device=device)
    renderer.render(save_images=False, save_video=True, output_dir=f'./rendered_views/{config_name}', object_name=object_name)

if __name__ == "__main__":
    main()
