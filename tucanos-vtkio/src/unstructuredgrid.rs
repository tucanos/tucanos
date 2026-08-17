use crate::{AppendedWriter, DataArray, FileType, Scalar};
use std::io::{Result, Write};

#[derive(Default)]
pub struct UnstructuredGridWriter<'a>(AppendedWriter<'a, UnstructuredGrid>);

impl<'a> UnstructuredGridWriter<'a> {
    pub const fn set_num_points(&mut self, n: usize) {
        self.0.file_type.number_of_points = n;
    }
    pub const fn set_num_cells(&mut self, n: usize) {
        self.0.file_type.number_of_cells = n;
    }
    pub fn add_points<T, IT>(&mut self, iterator: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        self.0.sections.insert(
            "Points",
            vec![DataArray::new(
                "Points",
                3,
                self.0.file_type.number_of_points,
                iterator,
            )],
        );
    }
    pub fn add_cells<TC, ITC, TO, ITO, TT, ITT>(
        &mut self,
        num_conn: usize,
        connectivity: ITC,
        offsets: ITO,
        types: ITT,
    ) where
        TC: Scalar + 'a,
        TO: Scalar + 'a,
        TT: Scalar + 'a,
        ITC: IntoIterator<Item = TC> + 'a,
        ITO: IntoIterator<Item = TO> + 'a,
        ITT: IntoIterator<Item = TT> + 'a,
    {
        self.0.sections.insert(
            "Cells",
            vec![
                DataArray::new("connectivity", 1, num_conn, connectivity),
                DataArray::new("offsets", 1, self.0.file_type.number_of_cells, offsets),
                DataArray::new("types", 1, self.0.file_type.number_of_cells, types),
            ],
        );
    }

    /// Add polyhedron faces. Must be called after `add_cells`.
    pub fn add_polyhedron_faces<T, ITF, ITO>(
        &mut self,
        num_conn: usize,
        faces: ITF,
        num_faces: usize,
        faceoffsets: ITO,
    ) where
        T: Scalar + 'a,
        ITF: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new("faces", 1, num_conn, faces));
        s.push(DataArray::new("faceoffsets", 1, num_faces, faceoffsets));
    }

    pub fn add_polyhedron_faces_v23<T, ITFC, ITFO>(
        &mut self,
        num_conn: usize,
        face_connectivity: ITFC,
        num_faces: usize,
        face_offsets: ITFO,
    ) where
        T: Scalar + 'a,
        ITFC: IntoIterator<Item = T> + 'a,
        ITFO: IntoIterator<Item = T> + 'a,
    {
        self.0.version = "2.3";
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new(
            "face_connectivity",
            1,
            num_conn,
            face_connectivity,
        ));
        s.push(DataArray::new("face_offsets", 1, num_faces, face_offsets));
    }

    pub fn add_polyhedron_face_map_v23<T, ITPF, ITPO>(
        &mut self,
        num_p_faces: usize,
        polyhedron_to_faces: ITPF,
        polyhedron_offsets: ITPO,
    ) where
        T: Scalar + 'a,
        ITPF: IntoIterator<Item = T> + 'a,
        ITPO: IntoIterator<Item = T> + 'a,
    {
        self.0.version = "2.3";
        let num_cells = self.0.file_type.number_of_cells;
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new(
            "polyhedron_to_faces",
            1,
            num_p_faces,
            polyhedron_to_faces,
        ));
        s.push(DataArray::new(
            "polyhedron_offsets",
            1,
            num_cells,
            polyhedron_offsets,
        ));
    }

    pub fn add_cell_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let num_cells = self.num_cells();
        self.0
            .sections
            .entry("CellData")
            .or_default()
            .push(DataArray::new(label, num_components, num_cells, values));
    }

    pub fn add_point_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let d = DataArray::new(
            label,
            num_components,
            self.0.file_type.number_of_points,
            values,
        );
        self.0.sections.entry("PointData").or_default().push(d);
    }

    const fn num_cells(&self) -> usize {
        self.0.file_type.number_of_cells
    }

    pub fn write(self, writer: &mut impl Write) -> Result<()> {
        self.0.write::<true>(writer)
    }
}

#[derive(Default)]
pub struct UnstructuredGrid {
    number_of_points: usize,
    number_of_cells: usize,
}

impl FileType for UnstructuredGrid {
    const NAME: &str = "UnstructuredGrid";

    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            r#"NumberOfPoints="{}" NumberOfCells="{}""#,
            self.number_of_points, self.number_of_cells
        )
    }
}
