//! Python bindings for extruded 2d meshes
use crate::{dual::PyDualMesh2d, mesh::PyMesh2d, poly::PyPolyMesh3d};
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::PyType,
};
use tmesh::{
    Tag, Vert3d,
    extruded::ExtrudedMesh2d,
    mesh::{Prism, Quadrangle, Simplex, Triangle},
};

/// Extruded 2d mesh
#[pyclass]
pub struct PyExtrudedMesh2d(pub(crate) ExtrudedMesh2d<usize>);

#[pymethods]
impl PyExtrudedMesh2d {
    /// Create a new mesh from coordinates, connectivities and tags
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        coords: PyReadonlyArray2<f64>,
        prisms: PyReadonlyArray2<usize>,
        prism_tags: PyReadonlyArray1<Tag>,
        tris: PyReadonlyArray2<usize>,
        tri_tags: PyReadonlyArray1<Tag>,
        quads: PyReadonlyArray2<usize>,
        quad_tags: PyReadonlyArray1<Tag>,
    ) -> PyResult<Self> {
        if coords.shape()[1] != 3 {
            return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
        }
        let n = prisms.shape()[0];
        if prisms.shape()[1] != 6 {
            return Err(PyValueError::new_err("Invalid dimension 1 for prisms"));
        }
        if prism_tags.shape()[0] != n {
            return Err(PyValueError::new_err("Invalid dimension 0 for prism_tags"));
        }

        let n = tris.shape()[0];
        if tris.shape()[1] != 3 {
            return Err(PyValueError::new_err("Invalid dimension 1 for tris"));
        }
        if tri_tags.shape()[0] != n {
            return Err(PyValueError::new_err("Invalid dimension 0 for tri_tags"));
        }

        let n = quads.shape()[0];
        if quads.shape()[1] != 4 {
            return Err(PyValueError::new_err("Invalid dimension 1 for quads"));
        }
        if quad_tags.shape()[0] != n {
            return Err(PyValueError::new_err("Invalid dimension 0 for quad_tags"));
        }

        let coords = coords.as_slice()?;
        let coords = coords
            .chunks(3)
            .map(|p| {
                let mut vx = Vert3d::zeros();
                vx.copy_from_slice(p);
                vx
            })
            .collect();

        let prisms = prisms.as_slice()?;
        let prisms = prisms
            .chunks(6)
            .map(|x| {
                let pri: [usize; 6] = x.try_into().unwrap();
                Prism::<usize>::from_iter(pri)
            })
            .collect();

        let tris = tris.as_slice()?;
        let tris = tris
            .chunks(3)
            .map(|x| {
                let tri: [usize; 3] = x.try_into().unwrap();
                Triangle::<usize>::from_iter(tri)
            })
            .collect();

        let quads = quads.as_slice()?;
        let quads = quads
            .chunks(4)
            .map(|x| {
                let quad: [usize; 4] = x.try_into().unwrap();
                Quadrangle::<usize>::from_iter(quad)
            })
            .collect();

        Ok(Self(ExtrudedMesh2d::new(
            coords,
            prisms,
            prism_tags.to_vec().unwrap(),
            tris,
            tri_tags.to_vec().unwrap(),
            quads,
            quad_tags.to_vec().unwrap(),
        )))
    }

    /// Extrude a 2d mesh
    #[classmethod]
    fn from_mesh2d(_cls: &Bound<'_, PyType>, msh: &PyMesh2d, h: f64) -> Self {
        Self(ExtrudedMesh2d::from_mesh2d(&msh.0, h))
    }

    /// Get the z=0 face as a 2d mesh
    fn to_mesh2d(&self) -> PyResult<PyMesh2d> {
        let msh = self
            .0
            .to_mesh2d()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyMesh2d(msh))
    }

    /// Number of vertices
    pub const fn n_verts(&self) -> usize {
        self.0.n_verts()
    }

    /// Get a copy of the vertices
    fn get_verts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut res = Vec::with_capacity(3 * self.0.n_verts());
        for v in self.0.verts() {
            for &x in v.as_slice() {
                res.push(x);
            }
        }
        PyArray::from_vec(py, res).reshape([self.0.n_verts(), 3])
    }

    /// Number of prisms
    const fn n_prisms(&self) -> usize {
        self.0.n_prisms()
    }

    /// Get a copy of the prisms
    fn get_prisms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
        PyArray::from_vec(py, self.0.prisms().copied().flatten().collect())
            .reshape([self.0.n_prisms(), 6])
    }

    /// Get a copy of the prism tags
    fn get_prism_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Tag>> {
        PyArray::from_vec(py, self.0.prism_tags().collect())
    }

    /// Number of triangles
    const fn n_tris(&self) -> usize {
        self.0.n_tris()
    }

    /// Get a copy of the triangles
    fn get_tris<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
        PyArray::from_vec(py, self.0.tris().copied().flatten().collect())
            .reshape([self.0.n_tris(), 3])
    }

    /// Get a copy of the triangle tags
    fn get_tri_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Tag>> {
        PyArray::from_vec(py, self.0.tri_tags().collect())
    }

    /// Number of quandrangles
    const fn n_quads(&self) -> usize {
        self.0.n_quads()
    }

    /// Get a copy of the quandrangles
    fn get_quads<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
        PyArray::from_vec(py, self.0.quads().copied().flatten().collect())
            .reshape([self.0.n_quads(), 4])
    }

    /// Get a copy of the quandrangle tags
    fn get_quad_tags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Tag>> {
        PyArray::from_vec(py, self.0.quad_tags().collect())
    }

    /// Export to a `.vtu` file
    fn write_vtk(&self, file_name: &str) -> PyResult<()> {
        self.0
            .write_vtk(file_name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyMesh2d {
    /// Extrude to 3d using 1 layer of prisms
    fn extrude(&self, h: f64) -> PyExtrudedMesh2d {
        let res = self.0.extrude(h);
        PyExtrudedMesh2d(res)
    }
}

#[pymethods]
impl PyDualMesh2d {
    /// Extrude to 3d using 1 layer of prisms
    fn extrude(&self, h: f64) -> PyPolyMesh3d {
        let res = self.0.extrude(h);
        PyPolyMesh3d(res)
    }
}
