use crate::{AppendedWriter, DataArray, FileType, Scalar};
use std::io::{Result, Write};

struct HyperTreeGrid {
    branch_factor: usize,
    transposed_root_indexing: bool,
    dimensions: [usize; 3],
    num_non_empty_trees: usize,
}

impl Default for HyperTreeGrid {
    fn default() -> Self {
        Self {
            branch_factor: 2,
            transposed_root_indexing: false,
            dimensions: [0; 3],
            num_non_empty_trees: 0,
        }
    }
}

impl FileType for HyperTreeGrid {
    const NAME: &str = "HyperTreeGrid";

    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            r#"BranchFactor="{}" TransposedRootIndexing="{}" Dimensions="{} {} {}""#,
            self.branch_factor,
            usize::from(self.transposed_root_indexing),
            self.dimensions[0],
            self.dimensions[1],
            self.dimensions[2]
        )
    }
}

/// Writer for VTK HyperTreeGrid XML files (`.htg`).
#[derive(Default)]
pub struct HyperTreeGridWriter<'a>(AppendedWriter<'a, HyperTreeGrid>);

impl<'a> HyperTreeGridWriter<'a> {
    /// Sets the grid dimensions along the X, Y, and Z axes.
    pub const fn set_dimensions(&mut self, dimensions: [usize; 3]) {
        self.0.file_type.dimensions = dimensions;
    }

    /// Sets the number of non-empty trees contained within the grid.
    pub const fn set_num_non_empty_trees(&mut self, num: usize) {
        self.0.file_type.num_non_empty_trees = num;
    }

    /// Sets the spatial coordinate arrays for the X, Y, and Z axes.
    pub fn set_grid<T, ITX, ITY, ITZ>(&mut self, itx: ITX, ity: ITY, itz: ITZ)
    where
        T: Scalar + 'a,
        ITX: IntoIterator<Item = T> + 'a,
        ITY: IntoIterator<Item = T> + 'a,
        ITZ: IntoIterator<Item = T> + 'a,
    {
        let [nx, ny, nz] = self.0.file_type.dimensions;
        self.0.sections.insert(
            "Grid",
            vec![
                DataArray::new("XCoordinates", 1, nx, itx),
                DataArray::new("YCoordinates", 1, ny, ity),
                DataArray::new("ZCoordinates", 1, nz, itz),
            ],
        );
    }

    /// Sets tree topology descriptors and depth arrays.
    pub fn set_trees<TD, ITD, ITN, ITT, ITDT>(
        &mut self,
        num_descriptor: usize,
        descriptor: ITD,
        num_verts_per_depth: ITN,
        tree_ids: ITT,
        depth_per_tree: ITDT,
    ) where
        TD: Scalar + 'a,
        ITD: IntoIterator<Item = TD> + 'a,
        ITN: IntoIterator<Item = i64> + 'a,
        ITT: IntoIterator<Item = i64> + 'a,
        ITDT: IntoIterator<Item = u32> + 'a,
    {
        self.0.version = "2.0";
        let n = self.0.file_type.num_non_empty_trees;
        self.0.sections.insert(
            "Trees",
            vec![
                DataArray::new_bits("Descriptors", 1, num_descriptor, descriptor),
                DataArray::new("NumberOfVerticesPerDepth", 1, n, num_verts_per_depth),
                DataArray::new("TreeIds", 1, n, tree_ids),
                DataArray::new("DepthPerTree", 1, n, depth_per_tree),
            ],
        );
    }

    pub fn add_cell_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let num_cells = self.0.file_type.num_non_empty_trees;
        self.0
            .sections
            .entry("CellData")
            .or_default()
            .push(DataArray::new(
                label,
                num_components,
                num_cells * num_components,
                values,
            ));
    }

    pub fn write(self, writer: &mut impl Write) -> Result<()> {
        self.0.write::<false>(writer)
    }
}
