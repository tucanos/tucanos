[![cargo test](https://github.com/tucanos/tucanos/actions/workflows/test.yml/badge.svg)](https://github.com/tucanos/tucanos/actions/workflows/test.yml)

# About

This repository contains libraries for operations on 2D and 3D simplex meshes, including
- [`tunacos`](https://github.com/tucanos/tucanos/tree/main/tucanos) for anisotropic mesh adaptation library based on [*Four-Dimensional Anisotropic Mesh Adaptation for
Spacetime Numerical Simulations* by
Philip Claude Caplan](https://www.cs.middlebury.edu/~pcaplan/docs/Caplan_2019_PhD.pdf). Tools for feature-based, geometry-based and mesh-implied metric computation are also provided, as well as for operations such as scaling and intersection.
- [`tmesh`](https://github.com/tucanos/tucanos/tree/main/tmesh) for general mesh operations, including creation from general elements, dual mesh computation, partitioning, ordering...

A Python wrapper can be found [here](https://github.com/tucanos/pytucanos).


## Building

Building the libraries requires a recent version of rust, that should be installed [as follows](https://www.rust-lang.org/tools/install)

### Dependencies not managed by Cargo

They are all optional.

* [`NLOpt`](https://github.com/stevengj/nlopt) can be used for smoothing, but the current implementation is quite inefficient
* [`metis`](https://github.com/KarypisLab/METIS) can be used to produce better quality mesh partitioning. The location of  `metis` may need to be declared using environment variables. This can be done in `.cargo/config.toml`, for example:
```toml
[env]
METISDIR="/path/to/metis_prefix"
```

### Features

Optional [cargo features](https://doc.rust-lang.org/cargo/reference/features.html) are:
    - `nlopt` 
    - `metis`
    - `coupe`

### Render doc

```
RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --no-deps --document-private-items
```

## Using from C with FFI

Tucanos provides a [C API](https://github.com/tucanos/tucanos/tree/main/tucanos-ffi) using
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

## Using from python

Python bindings for `tmesh` and `tucanos` are provided. 

## Remeshing benchmarks

### Reference codes

#### [MMG](https://github.com/MmgTools/mmg)

A minimal version can be installed with:

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_VTK=OFF -DUSE_ELAS=OFF -DUSE_SCOTCH=OFF
make -j$(nproc) install
```

NB: the performance will be better with scotch installed

#### [Omega\_h](https://github.com/sandialabs/omega_h)

Before building Omega`_h you shall ensure that
<https://github.com/sandialabs/omega_h/pull/408> is merged in you sources. Also
make sure that libMeshb is installed. A minimal version can then be installed
with:

```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOmega_h_USE_libMeshb=ON
make -j$(nproc) install
```

#### [Refine](https://github.com/nasa/refine)

A minimal version can be installed with:

```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc) install
```

#### [Avro](https://philipclaude.gitlab.io/avro/)

Detailed installation instructions are available [here](https://philipclaude.gitlab.io/avro/)

#### Docker / podman

A [`Dockerfile`](./pytucanos/benchmarks/Dockerfile) is included, and the image containing all the remeshers above may be build with e.g. as follows
```
cd pytucanos
podman build . -t remeshers
```

### Benchmarks

- [Isotropic remeshing in a 2D square domain
](pytucanos/benchmarks/square_iso/README.md)
- [Anisotropic remeshing in a 2D square domain
](pytucanos/benchmarks/square_linear/README.md)
- [Isotropic remeshing in a 3D cubic domain
](pytucanos/benchmarks/cube_iso/README.md)
- [Anisotropic remeshing in a 3D cubic domain
](pytucanos/benchmarks/cube_linear/README.md)
- [polar-1 benchmark from the UGAWG
](pytucanos/benchmarks/cube_cylinder/README.md)

