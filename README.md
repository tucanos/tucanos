[![cargo test](https://github.com/tucanos/tucanos/actions/workflows/test.yml/badge.svg)](https://github.com/tucanos/tucanos/actions/workflows/test.yml)

# About

This is a 2D and 3D anisotropic mesh adaptation library based on [*Four-Dimensional Anisotropic Mesh Adaptation for
Spacetime Numerical Simulations* by
Philip Claude Caplan](https://www.cs.middlebury.edu/~pcaplan/docs/Caplan_2019_PhD.pdf).

# Dependencies

## Required dependencied
* [libOL](https://github.com/LoicMarechal/libOL) is required and must be installed before building this library. It may be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j install
```
## Optional dependencies
* [libMeshb](https://github.com/LoicMarechal/libMeshb) is optional and used to enable benchmarks with reference codes. It may be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j install
```
* [NLOpt](https://github.com/stevengj/nlopt) can be used for smoothing, but the current implementation is quite inefficient
* Different LAPACK versions (Accelerate, MKL) may be used if available

# Building

* `libOL` (and optionally `libMeshb`) location may need to be declared using environment variables. This can be done in `.cargo/config.toml`, for example:
```toml
[env]
LIBOL_DIR="/path/to/libOL_prefix"
LIBMESHB_DIR="/path/to/libMeshb_prefix"
```
See <https://github.com/xgarnaud/libmeshb-sys> and <https://github.com/jeromerobert/marechal-libol-sys.git> for other possible environment variables.

* Optional features are
    - netlib / accelerate / openblas / intel-mkl to select the LAPACK version
    - nlopt to enable smoothing with NLOpt

Instead of specifying any of `< netlib | accelerate | openblas | intel-mkl>` you may specify the Lapack
implementation you prefer to use by adding this to the `.cargo/config.toml` file:

```toml
[env]
REMESH_LINK_DIRS="/usr/lib/x86_64-linux-gnu/openblas-pthread"
REMESH_LIBRARIES="openblasp-r0.3.21"
```

## Render doc

```
RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --no-deps --document-private-items
```