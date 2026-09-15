//! Generic zero-copy serializer for VTK XML files in Appended binary format.
//!
//! Provides traits and abstractions for formatting structured or unstructured
//! geometric primitives into raw binary VTK XML structures (`.vtp`).
use std::{
    collections::BTreeMap,
    io::{Result, Write},
    mem::size_of,
};

mod hypertreegrid;
mod multiblock;
mod polydata;
mod unstructuredgrid;
pub use hypertreegrid::HyperTreeGridWriter;
pub use multiblock::MultiBlockWriter;
pub use polydata::PolyDataWriter;
pub use unstructuredgrid::UnstructuredGridWriter;

/// Trait implemented by VTK file writers capable of exporting data to VTK XML format.
pub trait Writer {
    /// Associated file extension for the specific VTK dataset format (e.g., "vtu", "vtp", "vtm").
    const FILE_EXTENSION: &'static str;

    /// Writes the dataset content to the provided writer output stream.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if writing to the output stream fails.
    fn write(self, writer: &mut impl Write) -> Result<()>;
}

/// Trait implemented by specific VTK file structures (e.g., `PolyData`, `UnstructuredGrid`)
trait FileType {
    const NAME: &str;
    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()>;
}

struct AppendedWriter<'a, T: FileType> {
    file_type: T,
    version: &'static str,
    sections: BTreeMap<&'a str, Vec<DataArray<'a>>>,
    field_data: Vec<DataArray<'a>>,
}

impl<T: FileType + Default> Default for AppendedWriter<'_, T> {
    fn default() -> Self {
        Self {
            file_type: T::default(),
            version: "1.0",
            sections: BTreeMap::default(),
            field_data: Vec::default(),
        }
    }
}

impl<'a, T: FileType> AppendedWriter<'a, T> {
    fn write_section(
        writer: &mut impl Write,
        indent: &str,
        name: &str,
        arrays: &[DataArray<'a>],
        mut offset: usize,
    ) -> Result<usize> {
        if !arrays.is_empty() {
            writeln!(writer, "{indent}<{name}>")?;
            for a in arrays {
                writeln!(writer, "{indent}  {}", a.to_xml_tag(offset))?;
                offset += size_of::<u32>() + a.byte_len;
            }
            writeln!(writer, "{indent}</{name}>")?;
        }
        Ok(offset)
    }

    /// Writes the opening XML declaration and `<VTKFile>` root element header.
    fn write_header(&self, writer: &mut impl Write) -> Result<()> {
        let typ = T::NAME;
        let endianness = if cfg!(target_endian = "little") {
            "LittleEndian"
        } else {
            "BigEndian"
        };
        writeln!(
            writer,
            r#"<VTKFile type="{typ}" version="{}" byte_order="{endianness}" header_type="UInt32">"#,
            self.version
        )
    }

    /// Appends the raw binary data section and closes the root `<VTKFile>` tag.
    fn write_appended_data(&mut self, writer: &mut impl Write) -> Result<()> {
        write!(writer, "  <AppendedData encoding=\"raw\">\n   _")?;
        for section in self.sections.values_mut() {
            for array in section {
                array.write(writer)?;
            }
        }
        for array in &mut self.field_data {
            array.write(writer)?;
        }
        writeln!(writer, "\n  </AppendedData>\n</VTKFile>")
    }

    /// Writes optional field data arrays at the given indentation level.
    fn write_field_data(&self, indent: &str, writer: &mut impl Write, offset: usize) -> Result<()> {
        if !self.field_data.is_empty() {
            Self::write_section(
                writer,
                &format!("{indent}  "),
                "FieldData",
                &self.field_data,
                offset,
            )?;
        }
        Ok(())
    }

    fn write<const WITH_PIECE: bool>(mut self, writer: &mut impl Write) -> Result<()> {
        let typ = T::NAME;
        self.write_header(writer)?;
        let indent = if WITH_PIECE {
            writeln!(writer, "  <{typ}>")?;
            write!(writer, "    <Piece ")?;
            "      "
        } else {
            write!(writer, "  <{typ} ")?;
            "    "
        };
        self.file_type.write_piece_attributes(writer)?;
        writeln!(writer, ">")?;
        let mut offset = 0;
        for (name, arrays) in &self.sections {
            offset = Self::write_section(writer, indent, name, arrays, offset)?;
        }
        if WITH_PIECE {
            writeln!(writer, "    </Piece>")?;
        }
        self.write_field_data("  ", writer, offset)?;
        writeln!(writer, "  </{typ}>")?;
        self.write_appended_data(writer)
    }
}

struct DataArray<'a> {
    data_type: &'static str,
    name: String,
    number_of_tuples: usize,
    number_of_components: usize,
    byte_len: usize,
    data: Box<dyn WriteableIter + 'a>,
}

pub trait Scalar: Sized {
    const TYPE_NAME: &'static str;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()>;

    #[must_use]
    fn byte_len(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Scalar for &str {
    const TYPE_NAME: &'static str = "String";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(self.as_bytes())?;
        writer.write_all(b"\0")
    }

    fn byte_len(&self) -> usize {
        (*self).len() + 1
    }
}

#[cfg(target_pointer_width = "64")]
impl Scalar for usize {
    const TYPE_NAME: &'static str = "UInt64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i64 {
    const TYPE_NAME: &'static str = "Int64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i32 {
    const TYPE_NAME: &'static str = "Int32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i16 {
    const TYPE_NAME: &'static str = "Int16";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for u8 {
    const TYPE_NAME: &'static str = "UInt8";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i8 {
    const TYPE_NAME: &'static str = "Int8";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for u32 {
    const TYPE_NAME: &'static str = "UInt32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for f32 {
    const TYPE_NAME: &'static str = "Float32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for f64 {
    const TYPE_NAME: &'static str = "Float64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl<T: Scalar> Scalar for &T {
    const TYPE_NAME: &'static str = T::TYPE_NAME;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        (*self).write_ne_bytes(writer)
    }
}

trait WriteableIter {
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<usize>;
}

impl<I, T> WriteableIter for I
where
    I: Iterator<Item = T>,
    T: Scalar,
{
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<usize> {
        let mut size = 0;
        for item in self {
            item.write_ne_bytes(writer)?;
            size += item.byte_len();
        }
        Ok(size)
    }
}

impl<'a> DataArray<'a> {
    /// Creates a new bit-packed `DataArray` (`Bits` type).
    ///
    /// The input length `len` specifies the total number of bits, and total byte length
    /// is calculated as `ceil(len / 8)`.
    fn new_bits<IT>(name: &str, number_of_components: usize, len: usize, data: IT) -> Self
    where
        IT: IntoIterator + 'a,
        IT::Item: Scalar,
    {
        let number_of_tuples = len / number_of_components;
        let byte_len = len.div_ceil(8);
        let mut r =
            Self::with_byte_len(name, number_of_tuples, number_of_components, byte_len, data);
        r.data_type = "Bit";
        r
    }

    fn new<IT>(name: &str, number_of_components: usize, len: usize, data: IT) -> Self
    where
        IT: IntoIterator + 'a,
        IT::Item: Scalar,
    {
        let number_of_tuples = len / number_of_components;
        let byte_len = len * std::mem::size_of::<IT::Item>();
        Self::with_byte_len(name, number_of_tuples, number_of_components, byte_len, data)
    }

    fn with_byte_len<IT>(
        name: &str,
        number_of_tuples: usize,
        number_of_components: usize,
        byte_len: usize,
        data: IT,
    ) -> Self
    where
        IT: IntoIterator + 'a,
        IT::Item: Scalar,
    {
        Self {
            data_type: <IT::Item as Scalar>::TYPE_NAME,
            name: name.to_string(),
            number_of_tuples,
            number_of_components,
            data: Box::new(data.into_iter()),
            byte_len,
        }
    }

    #[must_use]
    fn to_xml_tag(&self, offset: usize) -> String {
        let comp_str = if self.number_of_components == 1 {
            String::new()
        } else {
            format!(r#" NumberOfComponents="{}""#, self.number_of_components)
        };

        format!(
            concat!(
                r#"<DataArray type="{}" Name="{}"{} "#,
                r#"NumberOfTuples="{}" format="appended" offset="{}"/>"#
            ),
            self.data_type, self.name, comp_str, self.number_of_tuples, offset,
        )
    }

    fn write<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        let len = self.byte_len as u32;
        writer.write_all(&len.to_ne_bytes())?;
        let writen_len = self.data.write_to(writer)?;
        assert_eq!(
            writen_len, self.byte_len,
            "Invalid DataArray length: name={}, number_of_tuples={}",
            self.name, self.number_of_tuples
        );
        Ok(())
    }
}
