use std::{
    fs::{File, create_dir_all},
    io::{self, BufWriter},
    path::{Path, PathBuf},
};

use crate::{AppendedWriter, FileType};

pub struct MultiBlockWriter<'a> {
    appended_writer: AppendedWriter<'a, MultiBlock>,
    path: &'a Path,
    block_names: Vec<(String, &'a str)>,
}

#[derive(Default)]
struct MultiBlock;

impl<'a> MultiBlockWriter<'a> {
    #[must_use]
    pub fn new(path: &'a Path) -> Self {
        let appended_writer: AppendedWriter<'_, MultiBlock> = AppendedWriter::default();
        Self {
            appended_writer,
            path,
            block_names: Vec::new(),
        }
    }

    /// Adds a dataset block to the multiblock structure and writes the sub-file to disk.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if creating output directories or writing the block file fails.
    pub fn add_block<W: crate::Writer>(
        &mut self,
        name: impl Into<String>,
        writer: W,
    ) -> io::Result<()> {
        let name = name.into();
        let dir_name = self.data_directory();
        let file_name = dir_name.join(&name).with_extension(W::FILE_EXTENSION);
        create_dir_all(&dir_name)?;
        writer.write(&mut BufWriter::new(File::create(file_name)?))?;
        self.block_names.push((name, W::FILE_EXTENSION));
        Ok(())
    }

    fn data_directory(&self) -> PathBuf {
        self.path.with_extension("")
    }

    fn relative_file_name(&self, name: &str, extension: &str) -> String {
        let dir_name = self.data_directory();
        let basename = dir_name
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data");
        format!("{basename}/{name}.{extension}")
    }

    /// Finalizes and writes the root `.vtm` index file to disk.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if file creation or writing fails.
    pub fn write(self) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(self.path)?);
        <Self as crate::Writer>::write(self, &mut w)
    }
}

impl FileType for MultiBlock {
    const NAME: &'static str = "vtkMultiBlockDataSet";

    fn write_piece_attributes(&self, _writer: &mut impl io::Write) -> io::Result<()> {
        Ok(())
    }
}

impl crate::Writer for MultiBlockWriter<'_> {
    const FILE_EXTENSION: &'static str = "vtm";

    fn write(mut self, writer: &mut impl io::Write) -> io::Result<()> {
        self.appended_writer.write_header(writer)?;
        writeln!(writer, "  <vtkMultiBlockDataSet>")?;
        for (index, (name, extension)) in self.block_names.iter().enumerate() {
            let file = self.relative_file_name(name, extension);
            writeln!(
                writer,
                r#"  <DataSet index="{index}" name="{name}" file="{file}"/>"#
            )?;
        }
        writeln!(writer, "  </vtkMultiBlockDataSet>")?;
        self.appended_writer.write_field_data("", writer, 0)?;
        self.appended_writer.write_appended_data(writer)
    }
}
