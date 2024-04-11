[![cargo test](https://github.com/tucanos/tucanos/actions/workflows/test.yml/badge.svg)](https://github.com/tucanos/tucanos/actions/workflows/test.yml)

# About

This is a 2D and 3D anisotropic mesh adaptation library based on [*Four-Dimensional Anisotropic Mesh Adaptation for
Spacetime Numerical Simulations* by
Philip Claude Caplan](https://www.cs.middlebury.edu/~pcaplan/docs/Caplan_2019_PhD.pdf).

A Python wrapper can be found [here](https://github.com/tucanos/pytucanos).

# Dependencies not managed by Cargo

They are all optional.

* [libOL](https://github.com/LoicMarechal/libOL) can be used to replace
  [parry](https://github.com/dimforge/parry)) for spatial indexing and
  projection. It may be installed with:
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
    - `parry` (enabled by default)
    - `nlopt` to enable smoothing with NLOpt
    - `meshb`
    - `libol` (enabled by default, disabled with `--no-default-features`)

Exactly one of `libol` or `parry` must be enabled.

## Render doc

```
RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --no-deps --document-private-items
```

# Using from C with FFI

Tucanos provides a C API using
[FFI](https://en.wikipedia.org/wiki/Foreign_function_interface). To be able to
use it first ensure the FFI wrapper is built using:

```
cargo build --workspace --release
```

The FFI files are:

```
target/release/tucanos.h
target/release/libtucanos.so
```

Here is an example of a C program using Tucanos:

```c
// build with gcc -Wl,-rpath=target/debug -Ltarget/debug -Itarget/debug test.c -ltucanos
#include <assert.h>
#include <stdio.h>
#include <tucanos.h>
int main(void) {
  double vertices[] = {0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.};
  uint32_t faces[] = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
  int16_t faces_tags[] = {1, 1, 1, 1};
  uint32_t elem[4] = {0, 1, 2, 3};
  int16_t elem_tag[1] = {1};
  double metric[4] = {0.1, 0.1, 0.1, 0.1};
  // Create an input mesh with a single tetrahedron
  tucanos_mesh33_t *mesh = tucanos_mesh33_new(4, vertices, 1, elem, elem_tag, 4, faces, faces_tags);
  tucanos_mesh32_t *boundary = tucanos_mesh33_boundary(mesh);
  tucanos_geom3d_t *geom = tucanos_geom3d_new(mesh, boundary);
  assert(geom != NULL);
  tucanos_remesher3diso_t *remesher = tucanos_remesher3diso_new(mesh, metric, geom);
  struct tucanos_params_t params;
  tucanos_params_init(&params);
  tucanos_remesher3diso_remesh(remesher, &params, geom);
  tucanos_mesh33_delete(mesh);
  tucanos_mesh33_t *new_mesh = tucanos_remesher3diso_tomesh(remesher, false);
  int num_verts = tucanos_mesh33_num_verts(new_mesh);
  printf("Number of vertices after remeshing: %d\n", num_verts);
  tucanos_mesh33_delete(new_mesh);
}
```
