use crate::{mesh::SimplexMesh, topo_elems::Elem, FieldLocation, FieldType, Idx, Mesh, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn write_array_formatted<T: std::fmt::Display>(
    file: &mut File,
    data: &[T],
    m: usize,
) -> Result<()> {
    assert_eq!(data.len() % m, 0);

    let n = data.len() / m;
    for i in 0..n {
        for j in 0..m {
            write!(file, "{} ", data[m * i + j])?;
        }
        writeln!(file)?;
    }

    Ok(())
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    fn write_xdmf_elems(&self, file: &mut File) -> Result<()> {
        let n_elems = self.n_elems();
        let etype = E::NAME;
        let edim = E::N_VERTS;

        write!(
            file,
            "            <Topology TopologyType=\"{etype}\" NumberOfElements=\"{n_elems}\">
                <DataItem Dimensions=\"{n_elems} {edim}\" NumberType=\"Int\">
"
        )?;
        write_array_formatted(file, &self.elems, E::N_VERTS as usize)?;
        write!(
            file,
            "                </DataItem>
            </Topology>
",
        )?;

        // Write the cell tags
        write!(
            file,
            "            <Attribute Name=\"Tag\" Center=\"Cell\">
                        <DataItem Dimensions=\"{n_elems}\">
        "
        )?;
        write_array_formatted(file, &self.etags, 1)?;
        write!(
            file,
            "                </DataItem>
                    </Attribute>
        "
        )?;

        Ok(())
    }

    fn write_xdmf_verts(&self, file: &mut File) -> Result<()> {
        let n_verts = self.n_verts();
        let gtype = if D == 2 { "XY" } else { "XYZ" };
        let vdim = D;
        write!(
            file,
            "            <Geometry GeometryType=\"{gtype}\">
                <DataItem Dimensions=\"{n_verts} {vdim}\">
"
        )?;
        write_array_formatted(file, &self.coords, D)?;
        write!(
            file,
            "                </DataItem>
            </Geometry>
"
        )?;

        Ok(())
    }

    fn write_xdmf_data(
        &self,
        file: &mut File,
        name: &str,
        loc: FieldLocation,
        vec: &[f64],
    ) -> Result<()> {
        let (center, n) = match loc {
            FieldLocation::Vertex => ("Node", self.n_verts()),
            FieldLocation::Element => ("Cell", self.n_elems()),
            FieldLocation::Constant => unreachable!(),
        };
        let m = vec.len() as Idx / n;
        let ftype = self.field_type(vec, loc).unwrap();
        match ftype {
            FieldType::Scalar | FieldType::Vector => {
                write!(
                    file,
                    "            <Attribute Name=\"{name}\" Center=\"{center}\">
                <DataItem Dimensions=\"{n} {m}\">
"
                )?;
                write_array_formatted(file, vec, m as usize)?;
                write!(
                    file,
                    "                </DataItem>
            </Attribute>
"
                )?;
            }
            FieldType::SymTensor => {
                for j in 0..m {
                    let mut tmp = Vec::with_capacity(n as usize);
                    tmp.extend((0..n).map(|i| vec[(m * i + j) as usize]));
                    write!(
                        file,
                        "            <Attribute Name=\"{name}_{j}\" Center=\"{center}\">
                <DataItem Dimensions=\"{n} {m}\">
"
                    )?;
                    write_array_formatted(file, &tmp, m as usize)?;
                    write!(
                        file,
                        "                </DataItem>
            </Attribute>
"
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Write an ascii xdmf file containing the mesh and the vertex and element data
    /// time : a time stamp that will be used to display animations in Paraview
    pub fn write_xdmf(
        &self,
        file_name: &str,
        time: Option<f64>,
        vertex_data: Option<HashMap<String, &[f64]>>,
        elem_data: Option<HashMap<String, &[f64]>>,
    ) -> Result<()> {
        let mut file = File::create(file_name)?;
        let time = time.unwrap_or(0.0);

        write!(
            file,
            "<?xml version=\"1.0\"?>
<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>
<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\">
    <Domain>
        <Grid Name=\"Elements\">
            <Time Type=\"Single\" Value=\"{time}\" />
"
        )?;

        self.write_xdmf_verts(&mut file)?;
        self.write_xdmf_elems(&mut file)?;
        if let Some(vertex_data) = vertex_data {
            for (name, data) in &vertex_data {
                self.write_xdmf_data(&mut file, name, FieldLocation::Vertex, data)?;
            }
        }
        if let Some(elem_data) = elem_data {
            for (name, data) in &elem_data {
                self.write_xdmf_data(&mut file, name, FieldLocation::Element, data)?;
            }
        }

        write!(
            file,
            "        </Grid>
    </Domain>
</Xdmf>
"
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };
    use std::fs::remove_file;

    #[test]
    fn test_xdmf_2d() -> Result<()> {
        let mesh = test_mesh_2d();
        mesh.write_xdmf("test_2d.xdmf", None, None, None)?;
        remove_file("test_2d.xdmf")?;
        Ok(())
    }

    #[test]
    fn test_xdmf_3d() -> Result<()> {
        let mesh = test_mesh_3d();
        mesh.write_xdmf("test_3d.xdmf", None, None, None)?;
        remove_file("test_3d.xdmf")?;
        Ok(())
    }
}
