## NeuOctree

Neural surface representation for ray tracing and feedforward construction.

**Paper:** TODO | **Pretrained models:** TODO | **Data:** TODO

### 1. Installation

Linux, an NVIDIA GPU, and Blender 3.x (for mesh preprocessing) are required.

```bash
conda create -n neuoct python=3.13 -y
conda activate neuoct
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[dev,preprocess,raytracing]'
```

Run the following commands from the repository root.

### 2. Optimize your own object

Prepare any GLB/glTF, OBJ, PLY, or STL mesh. Keep associated textures and material files alongside the mesh. The script renders supervision, holds out test views, and generates the point cloud and configuration:

```bash
python scripts/prepare_object.py --input /path/to/model.glb \
  --output data/my_object --object custom/my_object --blender /path/to/blender
```

Optimize **geometry and color**:

```bash
python scripts/train_object.py --config data/my_object/config.yaml \
  --object custom/my_object --output outputs/my_object --epochs 300
```

Add `--geometry-only` for geometry without color. Use `--feature-dim 16 --hidden-dim 128` to set capacity, or `--gpus 2` for two visible GPUs. Existing prepared datasets can be used directly through `--config` and any `--object category/object_id`.

Pack the model and evaluate on held-out views:

```bash
neuoct-pack-local-checkpoint --config data/my_object/config.yaml \
  --checkpoint outputs/my_object/custom-my_object.pt --object custom/my_object \
  --output outputs/my_object/inference_fp16.pt --storage-dtype fp16

neuoct-evaluate-object --config data/my_object/config.yaml \
  --checkpoint outputs/my_object/inference_fp16.pt --object custom/my_object \
  --output outputs/my_object/evaluation --data-split test --hit-threshold 0.4 --orient-normals
```

The paper's seven-object geometry presets remain available through `scripts/train_geometry.py --model Armadillo --tier 823`. They are optional presets; custom objects use `train_object.py`.

### 3. Feedforward training and inference

Download the processed data and weights (links pending), then set the data paths in `configs/feedforward.yaml`.

```bash
neuoct-train --config configs/feedforward.yaml SOLVER.gpu 0,1,2,3

neuoct-evaluate-feedforward --config configs/feedforward.yaml \
  --checkpoint checkpoints/feedforward.pth --output outputs/feedforward \
  --hit-threshold 0.4 --orient-normals
```

### 4. Ray tracing

Set the mesh, HDRI, and packed checkpoint paths in a scene YAML, then run:

```bash
python tools/neuoct_rt/render_scene.py --config configs/raytracing/chess_neuoct.yaml
```

For a quick check without external data or weights:

```bash
python scripts/smoke_pipeline.py --output outputs/smoke
python tools/neuoct_rt/smoke_render.py --output outputs/smoke.png --device cpu
```

### Citation

TODO
