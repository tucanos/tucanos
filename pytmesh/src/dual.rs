use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::{Bound, PyResult, Python, exceptions::PyRuntimeError, pyclass, pymethods};
use tmesh::{
    Tag,
    dual_mesh::{DualMesh, DualType},
    dual_mesh_2d::DualMesh2d,
    dual_mesh_3d::DualMesh3d,
    poly_mesh::PolyMesh,
};

use crate::mesh::{PyBoundaryMesh2d, PyBoundaryMesh3d, PyMesh2d, PyMesh3d};

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyDualType {
    Median,
    Barth,
}

macro_rules! create_dual_mesh {
    ($pyname: ident, $name: ident, $mesh: ident, $dim: expr, $cell_dim: expr, $face_dim: expr) => {
        #[pyclass]
        pub struct $pyname(pub(crate) $name);

        #[pymethods]
        impl $pyname {
            #[new]
            pub fn new(mesh: &$mesh, t: PyDualType) -> PyResult<Self> {
                let t = match t {
                    PyDualType::Median => DualType::Median,
                    PyDualType::Barth => DualType::Barth,
                };

                Ok(Self($name::new(&mesh.0, t)))
            }

            fn n_verts(&self) -> usize {
                self.0.n_verts()
            }

            fn get_verts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                PyArray::from_vec(py, self.0.seq_verts().flatten().cloned().collect())
                    .reshape([self.0.n_verts(), $dim])
            }

            fn n_elems(&self) -> usize {
                self.0.n_elems()
            }

            fn get_elems<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<bool>>,
            ) {
                let n = self.0.n_elems();
                let m = self.0.seq_elems().map(|x| x.len()).sum::<usize>();
                let mut ptr = Vec::with_capacity(n + 1);
                let mut e2f = Vec::with_capacity(m);
                let mut e2f_orient = Vec::with_capacity(m);

                ptr.push(0);
                for e in self.0.seq_elems() {
                    for &(f, o) in e {
                        e2f.push(f);
                        e2f_orient.push(o);
                    }
                    ptr.push(e2f.len());
                }

                (
                    PyArray::from_vec(py, ptr),
                    PyArray::from_vec(py, e2f),
                    PyArray::from_vec(py, e2f_orient),
                )
            }

            fn get_etags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.seq_etags().collect()))
            }

            fn n_faces(&self) -> usize {
                self.0.n_faces()
            }

            fn get_faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
                PyArray::from_vec(py, self.0.seq_faces().flatten().cloned().collect())
                    .reshape([self.0.n_faces(), $face_dim])
            }

            fn get_ftags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.seq_ftags().collect()))
            }

            fn elem_n_verts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
                let res = (0..self.0.n_elems())
                    .map(|i| self.0.elem_n_verts(i))
                    .collect::<Vec<_>>();
                PyArray::from_vec(py, res)
            }

            fn write_vtk(&self, file_name: &str) -> PyResult<()> {
                self.0
                    .write_vtk(file_name)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            fn check(&self) -> PyResult<()> {
                self.0
                    .check()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            fn elem<'py>(
                &self,
                py: Python<'py>,
                i: usize,
            ) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<bool>>) {
                let e = self.0.elem(i);
                let faces = e.iter().map(|&(i, _)| i).collect::<Vec<_>>();
                let orient = e.iter().map(|&(_, o)| o).collect::<Vec<_>>();
                (PyArray::from_vec(py, faces), PyArray::from_vec(py, orient))
            }

            fn face<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<usize>> {
                let f = self.0.face(i);
                PyArray::from_vec(py, f.to_vec())
            }
        }
    };
}

create_dual_mesh!(PyDualMesh2d, DualMesh2d, PyMesh2d, 2, 3, 2);
create_dual_mesh!(PyDualMesh3d, DualMesh3d, PyMesh3d, 3, 4, 3);

#[pymethods]
impl PyDualMesh2d {
    pub fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyBoundaryMesh2d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh2d(bdy), PyArray1::from_vec(py, ids))
    }
}

#[pymethods]
impl PyDualMesh3d {
    pub fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyBoundaryMesh3d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh3d(bdy), PyArray1::from_vec(py, ids))
    }
}
