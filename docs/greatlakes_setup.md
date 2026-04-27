# GreatLakes setup — GP-DLA pipeline

This document captures the environment setup that the pipeline was verified
against on UMich GreatLakes (account `cavestru0`). It supplements the generic
instructions in `README.md`.

## Login-node toolchain (verified 2026-04-25, gl3114)

- `gcc` 8.5.0
- `cmake` 3.26.5
- `git`, `wget`, `gawk`
- mamba/conda available via `/sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh`

## 1. Conda environment

```bash
mamba create -y -n gpdla python=3.11 pip
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla

pip install --upgrade pip
pip install numpy scipy astropy 'h5py>=3' matplotlib pytest \
            fitsio healpy scikit-learn pyyaml numba speclite
pip install desispec        # brings desiutil, desimodel, desitarget
pip install torch           # cu130 wheel; works on CPU on login node, GPU on a100/v100/gpu partition
```

Verified versions (2026-04-25):

| Package    | Version        |
|------------|----------------|
| python     | 3.11.15        |
| numpy      | 2.4.4          |
| scipy      | 1.17.1         |
| h5py       | 3.16.0         |
| astropy    | 7.2.0          |
| matplotlib | 3.10.8         |
| fitsio     | 1.3.0          |
| healpy     | 1.19.0         |
| desispec   | 0.70.0         |
| desiutil   | 3.6.1          |
| desimodel  | 0.20.0         |
| desitarget | 4.4.0          |
| torch      | 2.11.0+cu130   |

## 2. libcerf from source (REQUIRED — do not use a binary package)

The compiled C Voigt extension is significantly faster than the pure-Python
fallback. Build libcerf from source per the upstream instructions:

```bash
cd $HOME
git clone https://jugit.fz-juelich.de/mlz/libcerf.git
cd libcerf && mkdir -p build && cd build
cmake ..
make -j4
ctest         # 18 tests should pass
make install DESTDIR=$HOME/.local/
```

Installed shared libraries land at `$HOME/.local/usr/local/lib64/`.

## 3. Voigt C extension

```bash
cd /home/mfho/desi_gpy_dla_detection/gpy_dla_detection
cc -fPIC -shared -o _voigt.so ctypes_voigt.c \
    -I$HOME/.local/usr/local/include \
    -L$HOME/.local/usr/local/lib64 -lcerf
```

Add to your shell rc (one-time):

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

Smoke test:

```bash
LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH \
  python -c "from gpy_dla_detection.voigt_fast import VoigtProfile; \
             import numpy as np; \
             v = VoigtProfile(); \
             p = v.compute_voigt_profile(np.linspace(4200,4400,1000), 1e20.5, 2.5); \
             print('voigt OK, profile range', p.min(), p.max())"
```

## 4. Test suite

The CDDF + sample-generation tests do not require torch or desispec to be
loaded, and all 80 tests should pass:

```bash
LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH \
  python -m pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py \
                   tests/test_generate_samples.py -v
```

> **Note on numpy 2.x:** `np.trapz` was removed in numpy 2.0. We use a
> compatibility alias `_trapz = getattr(np, "trapezoid", np.trapz)` in
> `CDDF_analysis/cddf_mock.py` so the module works under both numpy 1.22+
> (NERSC stack) and numpy 2.x (this GreatLakes env). No behavioural change.

## 5. CUDA / GPU

GreatLakes login nodes (e.g. `gl3114`) do not expose GPUs.
`torch.cuda.is_available()` returns `False` there. To use a GPU, request one
in a SLURM job (`-p gpu --gres=gpu:1` or appropriate). The env's
`torch 2.11.0+cu130` wheel runs CUDA when scheduled on a GPU node.

## 6. Restricted-data path config

Real DESI LOA spectra paths are restricted. They are kept out of git via the
`private/` directory pattern in `.gitignore`. To set them on a new machine:

```bash
cp private/loa_paths.md.template private/loa_paths.md
$EDITOR private/loa_paths.md
```

`private/loa_paths.md` is gitignored automatically. The `.template` is
committed so the structure is self-documenting.
