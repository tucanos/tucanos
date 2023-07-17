use std::collections::HashMap;

use log::{debug, info};
use vtkio::{
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArrayBase, DataSet, ElementType,
        UnstructuredGridPiece, Version, VertexNumbers,
    },
    IOBuffer, Vtk,
};

use crate::{mesh::SimplexMesh, topo_elems::Elem, Mesh, Result};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn write_vtk(
        &self,
        file_name: &str,
        vertex_data: Option<HashMap<String, &[f64]>>,
        elem_data: Option<HashMap<String, &[f64]>>,
    ) -> Result<()> {
        info!("Write {file_name}");
        let connectivity = self.elems().flatten().map(|i| i as u64).collect();
        let offsets = (0..self.n_elems())
            .map(|i| (E::N_VERTS * (i + 1)) as u64)
            .collect();
        let cell_type = match E::N_VERTS {
            4 => CellType::Tetra,
            3 => CellType::Triangle,
            2 => CellType::PolyLine,
            _ => unreachable!(),
        };
        let mut point_data = Vec::new();
        if let Some(vertex_data) = vertex_data {
            for (name, arr) in vertex_data.iter() {
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
        cell_data.push(Attribute::DataArray(DataArrayBase {
            name: String::from("tag"),
            elem: ElementType::Scalars {
                num_comp: 1,
                lookup_table: None,
            },
            data: IOBuffer::I16(self.etags().collect()),
        }));

        if let Some(elem_data) = elem_data {
            for (name, arr) in elem_data.iter() {
                debug!("Write element data {name}");
                let num_comp = arr.len() / self.n_verts() as usize;
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
