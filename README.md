[![cargo test](https://github.com/tucanos/tucanos/actions/workflows/test.yml/badge.svg)](https://github.com/tucanos/tucanos/actions/workflows/test.yml)

# About

This is a 2D and 3D anisotropic mesh adaptation library based on [*Four-Dimensional Anisotropic Mesh Adaptation for
Spacetime Numerical Simulations* by
Philip Claude Caplan](https://www.cs.middlebury.edu/~pcaplan/docs/Caplan_2019_PhD.pdf).

# Dependencies not managed by Cargo

They are all optional.

* [libOL](https://github.com/LoicMarechal/libOL) is required and must be installed before building this library. It may be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j install
```
* [libMeshb](https://github.com/LoicMarechal/libMeshb) is optional and used to enable benchmarks with reference codes. It may be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j install
```
* [NLOpt](https://github.com/stevengj/nlopt) can be used for smoothing, but the current implementation is quite inefficient
* Different LAPACK versions (Accelerate, MKL) may be used if available

# Building

* `libOL` and `libMeshb` location may need to be declared using environment variables. This can be done in `.cargo/config.toml`, for example:
```toml
[env]
LIBOL_DIR="/path/to/libOL_prefix"
LIBMESHB_DIR="/path/to/libMeshb_prefix"
```
See <https://github.com/xgarnaud/libmeshb-sys> and <https://github.com/jeromerobert/marechal-libol-sys.git> for other possible environment variables.

* Optional [cargo features](https://doc.rust-lang.org/cargo/reference/features.html) are:
    - `libol` (enabled by default, disabled with `--no-default-features`)
    - `nlopt` to enable smoothing with NLOpt
    - `meshb`
    - `parry` (alternative to `libol`)

Exactly one of `libol` or `parry` must be enabled.

## Render doc

```
RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --no-deps --document-private-items
```
