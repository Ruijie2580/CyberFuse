# CyberFuse: A Deep Unfolding Network With Dynamic PID Control and Large-Kernel Attention for Hyperspectral Image Sharpening

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Each H5 file must contain `HRHS` of shape `(N, C, H, W)` and `HRMS` of shape
`(N, c, H, W)`. `LRHS` is optional — when absent it is generated on the fly by
Gaussian blur (kernel 7, sigma 2) followed by stride downsampling.

Place the datasets under `Dataset/` and set the paths in `configs/*.yaml`.

## Train

```bash
python trainer.py --config configs/chikusei.yaml
```

TensorBoard-style logging is handled by [SwanLab](https://swanlab.cn); set
`logging.use_swanlab: false` in the config to disable it.

## Test

```bash
python test.py --config configs/chikusei.yaml
```

Checkpoints are resolved automatically as
`{logging.save_dir}/{dataset.name}_best_model.pth`; pass `--ema` to load the
EMA weights or `--checkpoint` to point at a specific file.
