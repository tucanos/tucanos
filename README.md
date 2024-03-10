# About

Python bindings to [tucanos](https://github.com/tucanos/tucanos.git)

# Install

* Build *libMeshb* as explained in the [tucanos README](https://github.com/tucanos/tucanos#dependencies)
* [Install Rust](https://www.rust-lang.org/tools/install)
* Optionally enter your prefered `conda` or `venv` or `virtualenv`
* Run:

```bash
LIBMESHB_DIR=/path/to/libmeshb_prefix \
pip install git+https://github.com/tucanos/pytucanos.git`
```

# Benchmarks

## `.meshb/.solb` I/O

*libMeshb* is required to run the benchmarks with the reference codes below. It can be built as explained in the [tucanos README](https://github.com/tucanos/tucanos#dependencies).

## Reference codes

### [MMG](https://github.com/MmgTools/mmg)

A minimal version can be installed with:

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_VTK=OFF -DUSE_ELAS=OFF -DUSE_SCOTCH=OFF
make -j$(nproc) install
```

NB: the performance will be better with scotch installed

### [Omega\_h](https://github.com/sandialabs/omega_h)

Before building Omega`_h you shall ensure that
<https://github.com/sandialabs/omega_h/pull/408> is merged in you sources. Also
make sure that libMeshb is installed. A minimal version can then be installed
with:

```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOmega_h_USE_libMeshb=ON
make -j$(nproc) install
```

### [Refine](https://github.com/nasa/refine)

A minimal version can be installed with:

```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc) install
```

### [Avro](https://philipclaude.gitlab.io/avro/)

Detailed installation instructions are available [here](https://philipclaude.gitlab.io/avro/)

## Test cases

- [Isotropic remeshing in a 2D square domain
](benchmarks/square_iso/README.md)
- [Anisotropic remeshing in a 2D square domain
](benchmarks/square_linear/README.md)
- [Isotropic remeshing in a 3D cubic domain
](benchmarks/cube_iso/README.md)
- [Anisotropic remeshing in a 3D cubic domain
](benchmarks/cube_linear/README.md)
- [polar-1 benchmark from the UGAWG
](benchmarks/cube_cylinder/README.md)

