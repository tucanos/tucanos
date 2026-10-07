use crate::{
    Error, Result, Tag, Vertex,
    mesh::{Mesh, Simplex, SolutionLocation},
};
use parry2d_f64::utils::hashmap::HashMap;
use rust_hdf5::{H5File, H5Type};

pub struct HDF5File(H5File);

fn verts_to_raw<const D: usize>(verts: impl ExactSizeIterator<Item = Vertex<D>>) -> Vec<f64> {
    let mut res = Vec::with_capacity(verts.len() * D);
    for v in verts {
        res.extend(&v);
    }
    res
}

fn simplex_to_raw<C: Simplex>(simplices: impl ExactSizeIterator<Item = C>) -> Vec<u64> {
    let mut res = Vec::with_capacity(simplices.len() * C::N_VERTS);
    for s in simplices {
        res.extend(s.into_iter().map(|x| x as u64));
    }
    res
}

impl HDF5File {
    pub fn create(fname: &str) -> Result<Self> {
        Ok(Self(H5File::create(fname)?))
    }

    pub fn open(fname: &str) -> Result<Self> {
        Ok(Self(H5File::open(fname)?))
    }

    pub fn write_mesh<const D: usize, M: Mesh<D>>(&self, mesh: &M) -> Result<()> {
        let grp = self.0.create_group("mesh")?;

        let ds = grp.new_dataset::<u8>().shape([1]).create("order")?;
        ds.write_raw(&[<M::C as Simplex>::order()])?;

        let ds = grp
            .new_dataset::<f64>()
            .shape([mesh.n_verts(), D])
            .create("verts")?;
        ds.write_raw(&verts_to_raw(mesh.verts()))?;

        let ds = grp
            .new_dataset::<u64>()
            .shape([mesh.n_elems(), <M::C as Simplex>::N_VERTS])
            .create("elems")?;
        ds.write_raw(&simplex_to_raw(mesh.elems()))?;

        let ds = grp
            .new_dataset::<Tag>()
            .shape([mesh.n_elems()])
            .create("etags")?;
        ds.write_raw(&mesh.etags().collect::<Vec<_>>())?;

        let ds = grp
            .new_dataset::<u64>()
            .shape([
                mesh.n_faces(),
                <<M::C as Simplex>::FACE as Simplex>::N_VERTS,
            ])
            .create("faces")?;
        ds.write_raw(&simplex_to_raw(mesh.faces()))?;

        let ds = grp
            .new_dataset::<Tag>()
            .shape([mesh.n_faces()])
            .create("ftags")?;

        ds.write_raw(&mesh.ftags().collect::<Vec<_>>())?;

        self.0.flush()?;

        Ok(())
    }

    pub fn write_tag_names(&self, names: HashMap<Tag, String>) -> Result<()> {
        let ds = self.0.dataset_writer("mesh/ftags")?;

        for (tag, name) in names {
            let attr = ds.new_attr::<Tag>().shape(()).create(&name)?;
            attr.write_numeric(&tag)?;
        }

        self.0.flush()?;

        Ok(())
    }

    pub fn read_tag_names(&self) -> Result<HashMap<Tag, String>> {
        let ds = self.0.dataset("mesh/ftags")?;
        let mut names = HashMap::default();
        for name in ds.attr_names()? {
            let tag = ds.attr(&name)?.read_numeric::<Tag>()?;
            names.insert(tag, name);
        }
        Ok(names)
    }

    pub fn write_data<T: H5Type>(
        &self,
        name: &str,
        loc: &SolutionLocation,
        m: usize,
        data: &[T],
    ) -> Result<()> {
        let grp_name = match loc {
            SolutionLocation::Vertices => "vertex_data",
            SolutionLocation::Elements => "element_data",
            SolutionLocation::Faces => "face_data",
            SolutionLocation::Edges => "edge_data",
        };

        let grp = match self.0.create_group(grp_name) {
            Ok(grp) => grp,
            Err(_) => self.0.root_group().group(grp_name)?,
        };
        assert_eq!(data.len() % m, 0);
        let n = data.len() / m;
        let ds = grp.new_dataset::<T>().shape([n, m]).create(name)?;
        ds.write_raw(data)?;

        self.0.flush()?;
        Ok(())
    }

    pub fn read_mesh<const D: usize, M: Mesh<D>>(&self) -> Result<M> {
        let ds = self.0.dataset("mesh/order")?;
        let order = ds.read_raw::<u8>()?;
        if order.len() != 1 || order[0] != <M::C as Simplex>::order() {
            return Err(Error::from("Incompatible simplex order in HDF5 mesh"));
        }

        let ds = self.0.dataset("mesh/verts")?;
        let shape = ds.shape();
        if shape.len() != 2 || shape[1] != D {
            return Err(Error::from("Invalid verts dataset shape"));
        }
        let verts_raw = ds.read_raw::<f64>()?;
        let verts = verts_raw
            .chunks(D)
            .map(Vertex::<D>::from_column_slice)
            .collect::<Vec<_>>();
        if verts.len() != shape[0] {
            return Err(Error::from("Invalid verts dataset size"));
        }

        let ds = self.0.dataset("mesh/elems")?;
        let shape = ds.shape();
        if shape.len() != 2 || shape[1] != <M::C as Simplex>::N_VERTS {
            return Err(Error::from("Invalid elems dataset shape"));
        }
        let elems_raw = ds.read_raw::<u64>()?;
        let elems = elems_raw
            .chunks(<M::C as Simplex>::N_VERTS)
            .map(|c| <M::C as Simplex>::from_iter(c.iter().map(|&x| x as usize)))
            .collect::<Vec<_>>();
        if elems.len() != shape[0] {
            return Err(Error::from("Invalid elems dataset size"));
        }

        let ds = self.0.dataset("mesh/etags")?;
        let etags = ds.read_raw::<Tag>()?;
        if etags.len() != elems.len() {
            return Err(Error::from("Inconsistent elems/etags dataset sizes"));
        }

        let ds = self.0.dataset("mesh/faces")?;
        let shape = ds.shape();
        if shape.len() != 2 || shape[1] != <<M::C as Simplex>::FACE as Simplex>::N_VERTS {
            return Err(Error::from("Invalid faces dataset shape"));
        }
        let faces_raw = ds.read_raw::<u64>()?;
        let faces = faces_raw
            .chunks(<<M::C as Simplex>::FACE as Simplex>::N_VERTS)
            .map(|c| <M::C as Simplex>::FACE::from_iter(c.iter().map(|&x| x as usize)))
            .collect::<Vec<_>>();
        if faces.len() != shape[0] {
            return Err(Error::from("Invalid faces dataset size"));
        }

        let ds = self.0.dataset("mesh/ftags")?;
        let ftags = ds.read_raw::<Tag>()?;
        if ftags.len() != faces.len() {
            return Err(Error::from("Inconsistent faces/ftags dataset sizes"));
        }

        Ok(M::new(&verts, &elems, &etags, &faces, &ftags))
    }

    pub fn read_data<T: H5Type>(
        &self,
        name: &str,
        loc: &SolutionLocation,
    ) -> Result<(Vec<T>, usize)> {
        let grp_name = match loc {
            SolutionLocation::Vertices => "vertex_data",
            SolutionLocation::Elements => "element_data",
            SolutionLocation::Faces => "face_data",
            SolutionLocation::Edges => "edge_data",
        };

        let ds = self.0.dataset(&format!("{grp_name}/{name}"))?;
        let shape = ds.shape();
        let m = match shape.as_slice() {
            [_, m] => *m,
            [_] => 1,
            _ => return Err(Error::from("Invalid data dataset shape")),
        };
        let data = ds.read_raw::<T>()?;

        Ok((data, m))
    }
}

#[cfg(test)]
mod tests {
    use super::HashMap;
    use crate::{
        Result,
        io::hdf5_io::HDF5File,
        mesh::{Mesh, Mesh2d, Mesh3d, SolutionLocation, box_mesh, rectangle_mesh},
    };

    #[test]
    fn test_2d() -> Result<()> {
        let msh: Mesh2d = rectangle_mesh(1.0, 10, 1.0, 20);
        let fname = "test_2d.h2";

        let f = HDF5File::create(fname)?;
        f.write_mesh(&msh)?;
        let mut tag_names = HashMap::default();
        tag_names.insert(1, "bottom".to_string());
        tag_names.insert(2, "right".to_string());
        tag_names.insert(3, "top".to_string());
        tag_names.insert(4, "left".to_string());
        f.write_tag_names(tag_names.clone())?;
        let data0 = (0..msh.n_verts()).map(|i| i as f64).collect::<Vec<_>>();
        f.write_data("f0", &SolutionLocation::Vertices, 1, &data0)?;
        let data1 = (0..2 * msh.n_verts())
            .map(|i| (i as f64) * 0.5)
            .collect::<Vec<_>>();
        f.write_data("f1", &SolutionLocation::Vertices, 2, &data1)?;
        drop(f);

        let f = HDF5File::open(fname)?;
        let new_msh = f.read_mesh::<2, Mesh2d>()?;
        let new_tag_names = f.read_tag_names()?;
        let (new_data0, m0) = f.read_data::<f64>("f0", &SolutionLocation::Vertices)?;
        let (new_data1, m1) = f.read_data::<f64>("f1", &SolutionLocation::Vertices)?;

        msh.check_equals(&new_msh, 1e-12)?;
        assert_eq!(new_tag_names, tag_names);
        assert_eq!(m0, 1);
        assert_eq!(new_data0, data0);
        assert_eq!(m1, 2);
        assert_eq!(new_data1, data1);
        std::fs::remove_file(fname)?;

        Ok(())
    }

    #[test]
    fn test_3d() -> Result<()> {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 20, 1.0, 30);
        let fname = "test_3d.h2";

        let f = HDF5File::create(fname)?;
        f.write_mesh(&msh)?;
        let mut tag_names = HashMap::default();
        tag_names.insert(1, "xmin".to_string());
        tag_names.insert(2, "xmax".to_string());
        tag_names.insert(3, "ymin".to_string());
        tag_names.insert(4, "ymax".to_string());
        tag_names.insert(5, "zmin".to_string());
        tag_names.insert(6, "zmax".to_string());
        f.write_tag_names(tag_names.clone())?;
        let data = (0..3 * msh.n_elems())
            .map(|i| ((i as f64) + 1.0) / 7.0)
            .collect::<Vec<_>>();
        f.write_data("e0", &SolutionLocation::Elements, 3, &data)?;
        drop(f);

        let f = HDF5File::open(fname)?;
        let new_msh = f.read_mesh::<3, Mesh3d>()?;
        let new_tag_names = f.read_tag_names()?;
        let (new_data, m) = f.read_data::<f64>("e0", &SolutionLocation::Elements)?;

        msh.check_equals(&new_msh, 1e-12)?;
        assert_eq!(new_tag_names, tag_names);
        assert_eq!(m, 3);
        assert_eq!(new_data, data);
        std::fs::remove_file(fname)?;

        Ok(())
    }
}
