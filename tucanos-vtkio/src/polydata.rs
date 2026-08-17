use crate::{AppendedWriter, DataArray, FileType, Scalar};
use std::io::{Result, Write};

#[derive(Default)]
struct PolyData {
    points: usize,
    lines: usize,
    verts: usize,
}

impl FileType for PolyData {
    const NAME: &str = "PolyData";

    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            r#"NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="{}""#,
            self.points, self.verts, self.lines
        )
    }
}

/// Helper to write VTK `PolyData` in appended format.
#[derive(Default)]
pub struct PolyDataWriter<'a>(AppendedWriter<'a, PolyData>);

impl<'a> PolyDataWriter<'a> {
    /// Sets the total number of points in the dataset.
    pub const fn set_num_points(&mut self, n: usize) {
        self.0.file_type.points = n;
    }

    /// Sets the total number of line cells in the dataset.
    pub const fn set_num_lines(&mut self, n: usize) {
        self.0.file_type.lines = n;
    }

    /// Sets the total number of vertex cells in the dataset.
    pub const fn set_num_verts(&mut self, n: usize) {
        self.0.file_type.verts = n;
    }

    /// Adds 3D point coordinates to the `.vtp` dataset.
    ///
    /// # Arguments
    /// * `iterator` - Iterator yielding coordinate values for all points.
    ///   The total number of scalar items must equal `3 * set_num_points`.
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
                self.0.file_type.points,
                iterator,
            )],
        );
    }
    fn add_elems<T, ITO, ITC>(
        &mut self,
        num_conn: usize,
        connectivity: ITC,
        num: usize,
        offsets: ITO,
        section: &'a str,
    ) where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.0.sections.insert(
            section,
            vec![
                DataArray::new("connectivity", 1, num_conn, connectivity),
                DataArray::new("offsets", 1, num, offsets),
            ],
        );
    }

    /// Adds vertex elements (point cells) to the `.vtp` file.
    ///
    /// # Arguments
    /// * `num_conn` - Total number of connectivity indices across all vertex cells.
    /// * `connectivity` - Iterator over point indices referenced by the vertices.
    /// * `offsets` - Iterator over cumulative end-index offsets for each vertex cell.
    pub fn add_verts<T, ITO, ITC>(&mut self, num_conn: usize, connectivity: ITC, offsets: ITO)
    where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.add_elems(
            num_conn,
            connectivity,
            self.0.file_type.verts,
            offsets,
            "Verts",
        );
    }

    /// Adds polyline elements to the `.vtp` file.
    ///
    /// Supports multi-segment polylines using explicit connectivity offsets.
    ///
    /// # Arguments
    /// * `num_conn` - Total number of connectivity indices across all line cells.
    /// * `connectivity` - Iterator over point indices forming the lines.
    /// * `offsets` - Iterator yielding cumulative end-index offsets for each polyline cell.
    pub fn add_lines<T, ITO, ITC>(&mut self, num_conn: usize, connectivity: ITC, offsets: ITO)
    where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.add_elems(
            num_conn,
            connectivity,
            self.0.file_type.lines,
            offsets,
            "Lines",
        );
    }

    /// Adds a `CellData` field array to the dataset.
    ///
    /// In VTK PolyData (`.vtp`), cell attributes for all cell types are concatenated
    /// into a single contiguous array. The values in `values` **must** be ordered
    /// sequentially by cell type in the following order:
    ///
    /// 1. **Verts** (Vertices / PolyVertices)
    /// 2. **Lines** (Lines / PolyLines)
    /// 3. **Polys** (Triangles, Quads, Polygons)
    /// 4. **Strips** (Triangle Strips)
    ///
    /// Within each cell type, data points must match the insertion order of the cells.
    /// If `num_components > 1` (e.g. 3 for 3D vectors), components for a single cell
    /// are packed contiguously before moving to the next cell.
    ///
    /// # Arguments
    ///
    /// * `label` - Name of the field array
    /// * `num_components` - Number of components per cell (e.g., `1` for scalar, `3` for 3D vector).
    /// * `values` - Iterator yielding scalar values for all cells combined.
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

    const fn num_cells(&self) -> usize {
        self.0.file_type.verts + self.0.file_type.lines
    }

    pub fn write(self, writer: &mut impl Write) -> Result<()> {
        self.0.write::<true>(writer)
    }
}
