# About

Python bindings to [tucanos](https://github.com/tucanos/tucanos.git)

## Benchmarks

### Reference codes

#### [MMG](https://github.com/MmgTools/mmg)

A minimal version can be installed with:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_VTK=OFF -DUSE_ELAS=OFF -DUSE_SCOTCH=OFF
make -j install
```
NB: the performance will be better with scotch installed

#### [Omega\_h](https://github.com/sandialabs/omega_h)

Before building Omega_h you shall ensure that https://github.com/sandialabs/omega_h/pull/408 is merged in you sources. Also make sure that libMeshb is installed. A minimal version can then be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOmega_h_USE_libMeshb=ON
make -j install
```


#### [Refine](https://github.com/nasa/refine)

A minimal version can be installed with:
```
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j install
```

### Test cases

- [Isotropic remeshing in a 2D square domain
](benchmarks/square_iso/README.md)
- [Anisotropic remeshing in a 2D square domain
](benchmarks/square_linear/README.md)
- [Isotropic remeshing in a 3D cubic domain
](benchmarks/cube_iso/README.md)
- [Anisotropic remeshing in a 3D cubic domain
](benchmarks/cube_linear/README.md)

