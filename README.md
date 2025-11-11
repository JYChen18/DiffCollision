# Differentiable Collision Detection (DiffCollision)

Official implementation for [Robust Differentiable Collision Detection for General Objects](https://arxiv.org/abs/2511.06267).

## Overview

**DiffCollision** is a robust and efficient differentiable collision detection framework for general 3D objects, supporting both convex and concave geometries.

It builds on the high-performance [Coal](https://github.com/coal-library/coal) library for forward collision queries and integrates with **PyTorch autograd** for gradient computation, enabling seamless use in optimization and learning pipelines.

### Key Features

- **Parallel multi-mesh batching** — Efficiently handles multiple mesh pairs and batches in parallel, supporting up to around 0.5 billion convex piece pairs on a single 24 GB RTX 4090 GPU.
- **Unified plug-and-play API** — A single PyTorch interface with multiple differentiable backends: `RS1Dist`(ours, recommended), `RS1Dir`, `RS0`, `FD`, `Analytical`.
- **Strong performance** — Achieves >90% mm-level accuracy for convex and >70% for concave objects (DexGraspNet & Objaverse) in our main optimization benchmark.


## Example
```python
import torch
from diffcollision import DiffCollision, DCMesh

# Load meshes and define mesh pairs for collision detection
meshes = [DCMesh.from_file(f"obj{i}.obj", scale=0.1) for i in range(7)]
collision_pairs = [[0, 1], [0, 3], [3, 2], [1, 6], [6, 5]]  # total 5 mesh pairs

# Initialize differentiable collision module
diffcoll = DiffCollision(meshes, collision_pairs, method="RS1Dist")

# Batched transformation matrices for each mesh, shape (batch, n_mesh, 4, 4)
transforms = torch.eye(4, requires_grad=True)[None, None].expand(13, 7, 4, 4)

# Forward & backward. wp1/wp2: witness points, shape (batch, n_pair, 3)
result = diffcoll.forward(transforms)   # including wp1/wp2, normal, sdf, etc.
((result.wp1 - result.wp2) ** 2).sum().backward()   
```
For more examples:
- `tests/test_sphere.py` — basic usage & installation verification
- `tests/test_multi_mesh.py` — dynamic updates of collision pairs at runtime


## Installation
Create and activate a Conda environment with the `coal` dependency
```bash
conda create -n dcd python=3.10 coal -c conda-forge
conda activate dcd
```
Then install **DiffCollision**:
```bash
pip install -e .              # Core library only
pip install -e '.[examples]'  # Include dependencies for examples
```

## Reproducing Paper Results
### 1. Prepare object assets

Download the preprocessed assets from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy):
- `DGN_5k_processed.zip`
- `objaverse_5k_processed.zip` 

Unzip and organize as:
```
examples/assets/object
├── DGN_5k/processed_data
│   ├── core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
│   └── ...
└── objaverse_5k/processed_data
    ├── 0a3d04fec6544bb697e14d0c473077ed
    └── ...

```

### 2. Run the main optimization task
```bash
CUDA_VISIBLE_DEVICES=0 bash examples/scripts/baseline/ours.sh
```
To debug and visualization,
```bash
python examples/main.py exp=debug n_tp=10 vis=True n_prob=1 prob_rand=None 'obj=[core_jar_5a4af043ba70f9682db9fca4b68095, sem_TissueBox_fd9c40cd2ff2aab4a843bb865a04c01a]' 'scale=[0.08238040130667001, 0.04604694495201166]' 
```

### 3. Run the grasp refinement task
```bash
python examples/grasp.py exp=grasp vis=True
```


## License

This work is licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful, please consider to cite:

```bibtex
@article{chen2025robust,
    title={Robust Differentiable Collision Detection for General Objects}, 
    author={Chen, Jiayi and Zhao, Wei and Ruan, Liangwang and Chen, Baoquan and Wang, He},
    journal={arXiv preprint arXiv:2511.06267},
    year={2025}
}
```