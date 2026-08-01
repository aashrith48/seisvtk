# seisvtk

3D seismic visualization with VTK.

Runs on CPU or GPU — compute takes a `device` that falls back to CPU when no CUDA
device is present.

## Status

Early development.

## Installation

```bash
conda activate seis_env
pip install -e .
```

PyTorch is hardware-specific and best installed before the package. Blackwell cards
(RTX 50-series, `sm_120`) need a CUDA 12.8+ build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # Blackwell
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Ada / Ampere / Hopper
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only
```

Check the build is native for your card:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

## Layout

```
src/seisvtk/    # package source
```

## License

[MIT](LICENSE) © 2026 Aashrit
