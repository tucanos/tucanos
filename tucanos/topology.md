# Mesh topology


## Tags

Tags are signed integers. Negative tags indicate entities that should not be remeshed.

## Elements

All elements (dimension $D$) have tags.

## Tags

Elements of dimension $d < D$ are tagged iif any of the following is true
- they belong to only one tagged parent element
- they belong to 2 tagged parents with different tags
- they belong to 3 or more tagged parents
- ~~(relevant for vertices in 3d only?)~~ they belong to 2 *or more* tagged ~~grand parents (=faces)~~ ancestors * of dimension $d' > d$* with different tags
- ~~(relevant for vertices in 3d only?) they belong to 3 or more tagged grand parents (=faces)~~

Tagged faces (dimension $D - 1$) must be explicitely in the mesh.

## Tag hierarchy


## Vertex tags

