[![cargo test](https://github.com/tucanos/tucanos/actions/workflows/test.yml/badge.svg)](https://github.com/tucanos/tucanos/actions/workflows/test.yml)

# About

This is a 2D and 3D anisotropic mesh adaptation library based on [*Four-Dimensional Anisotropic Mesh Adaptation for
Spacetime Numerical Simulations* by
Philip Claude Caplan](https://www.cs.middlebury.edu/~pcaplan/docs/Caplan_2019_PhD.pdf).

A Python wrapper can be found [here](https://github.com/tucanos/pytucanos).

# Dependencies not managed by Cargo

They are all optional.

* [NLOpt](https://github.com/stevengj/nlopt) can be used for smoothing, but the current implementation is quite inefficient
* Different LAPACK versions (Accelerate, MKL) may be used if available

# Building

* `metis` and `scotch` locations may need to be declared using environment variables. This can be done in `.cargo/config.toml`, for example:
```toml
[env]
METISDIR="/path/to/metis_prefix"
SCOTCHDIR="/path/to/scotch_prefix"
```

* Optional [cargo features](https://doc.rust-lang.org/cargo/reference/features.html) are:
    - `nlopt` to enable smoothing with NLOpt
    - `metis`
    - `scotch`

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
// build with gcc -Wl,-rpath=target/release -Ltarget/release -Itarget/release test.c -ltucanos
// or you prefered build system
#include <assert.h>
#include <stdio.h>
#include <tucanos.h>
int main(void) {
  tucanos_init_log(); // use export RUST_LOG=debug to log more
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
  tucanos_remesher3diso_remesh(remesher, geom);
  tucanos_mesh33_delete(mesh);
  tucanos_geom3d_delete(geom);
  tucanos_mesh33_t *new_mesh = tucanos_remesher3diso_tomesh(remesher, false);
  tucanos_remesher3diso_delete(remesher);
  int num_verts = tucanos_mesh33_num_verts(new_mesh);
  printf("Number of vertices after remeshing: %d\n", num_verts);
  tucanos_mesh33_delete(new_mesh);
}
```
