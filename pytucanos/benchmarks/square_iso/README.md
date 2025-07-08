# Isotropic remeshing in a square

## Configuration

The geometry is an unit square with two different tags above and below the diagonal, so there is an internal surface.

The target isotropic cell size is 
$$h = h_{min} + (h_{max} - h_{min})  (
            1 - exp(-((x - 0.5)^2 + (y - 0.25)^2) / 0.25^2)
        )$$
with $h_{min} = 0.01$ and $h_{max} = 0.3$

![Config](mesh.png)

## Start mesh

The initial mesh only contains two triangles, so 5 iterations are required to reach the target cell sizes.

## Results after 5 iterations

![quality](quality.png)
![perfo](perfo.png)


