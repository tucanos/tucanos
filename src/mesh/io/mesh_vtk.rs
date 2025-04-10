use crate::{
    Result,
    mesh::{Elem, SimplexMesh},
};
use log::debug;
use std::collections::HashMap;
use vtkio::{
    IOBuffer, Vtk,
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArrayBase, DataSet, ElementType,
        UnstructuredGridPiece, Version, VertexNumbers,
    },
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn write_vtk(
        &self,
        file_name: &str,
        vertex_data: Option<HashMap<String, &[f64]>>,
        elem_data: Option<HashMap<String, &[f64]>>,
    ) -> Result<()> {
        debug!("Write {file_name}");
        let connectivity = self.elems().flatten().map(u64::from).collect();
        let offsets = (0..self.n_elems())
            .map(|i| u64::from(E::N_VERTS * (i + 1)))
            .collect();
        let cell_type = match E::N_VERTS {
            4 => CellType::Tetra,
            3 => CellType::Triangle,
            2 => CellType::PolyLine,
            _ => unreachable!(),
        };
        let mut point_data = Vec::new();
        if let Some(vertex_data) = vertex_data {
            for (name, arr) in &vertex_data {
                debug!("Write vertex data {name}");
                let num_comp = arr.len() / self.n_verts() as usize;
                point_data.push(Attribute::DataArray(DataArrayBase {
                    name: name.to_string(),
                    elem: ElementType::Scalars {
                        num_comp: num_comp as u32,
                        lookup_table: None,
                    },
                    data: IOBuffer::F64(arr.to_vec()),
                }));
            }
        }

        let mut cell_data = Vec::new();
        #[cfg(feature = "64bit-tags")]
        let tag_data = IOBuffer::I64(self.etags().collect());
        #[cfg(feature = "32bit-tags")]
        let tag_data = IOBuffer::I32(self.etags().collect());
        #[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
        let tag_data = IOBuffer::I16(self.etags().collect());
        cell_data.push(Attribute::DataArray(DataArrayBase {
            name: String::from("tag"),
            elem: ElementType::Scalars {
                num_comp: 1,
                lookup_table: None,
            },
            data: tag_data,
        }));

        if let Some(elem_data) = elem_data {
            for (name, arr) in &elem_data {
                debug!("Write element data {name}");
                let num_comp = arr.len() / self.n_elems() as usize;
                cell_data.push(Attribute::DataArray(DataArrayBase {
                    name: name.to_string(),
                    elem: ElementType::Scalars {
                        num_comp: num_comp as u32,
                        lookup_table: None,
                    },
                    data: IOBuffer::F64(arr.to_vec()),
                }));
            }
        }

        let mut coords = Vec::with_capacity(3 * self.n_verts() as usize);
        self.verts().for_each(|p| {
            for i in 0..D {
                coords.push(p[i]);
            }
            if D < 3 {
                coords.resize(coords.len() + 3 - D, 0.0);
            }
        });

        let vtk = Vtk {
            version: Version { major: 1, minor: 0 },
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(coords),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity,
                        offsets,
                    },
                    types: vec![cell_type; self.n_elems() as usize],
                },
                data: Attributes {
                    point: point_data,
                    cell: cell_data,
                },
            }),
        };

        vtk.export(file_name)?;

        Ok(())
    }
}
