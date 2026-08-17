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
mod polydata;
mod unstructuredgrid;
pub use hypertreegrid::HyperTreeGridWriter;
pub use polydata::PolyDataWriter;
pub use unstructuredgrid::UnstructuredGridWriter;

/// Trait implemented by specific VTK file structures (e.g., `PolyData`, `UnstructuredGrid`)
trait FileType {
    const NAME: &str;
    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()>;
}

struct AppendedWriter<'a, T: FileType> {
    file_type: T,
    version: &'static str,
    sections: BTreeMap<&'a str, Vec<DataArray<'a>>>,
}

impl<T: FileType + Default> Default for AppendedWriter<'_, T> {
    fn default() -> Self {
        Self {
            file_type: T::default(),
            version: "1.0",
            sections: BTreeMap::default(),
        }
    }
}

impl<'a, T: FileType> AppendedWriter<'a, T> {
    fn write<const WITH_PIECE: bool>(mut self, writer: &mut impl Write) -> Result<()> {
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
        )?;
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

        let mut write_section = |name: &str, arrays: &[DataArray<'a>]| -> Result<()> {
            if !arrays.is_empty() {
                writeln!(writer, "{indent}<{name}>")?;
                for a in arrays {
                    writeln!(writer, "{indent}  {}", a.to_xml_tag(offset))?;
                    offset += size_of::<u32>() + a.byte_len;
                }
                writeln!(writer, "{indent}</{name}>")?;
            }
            Ok(())
        };

        for (name, arrays) in &self.sections {
            write_section(name, arrays)?;
        }
        if WITH_PIECE {
            writeln!(writer, "    </Piece>")?;
        }
        write!(
            writer,
            "  </{typ}>\n  <AppendedData encoding=\"raw\">\n   _"
        )?;
        for section in self.sections.values_mut() {
            for array in section {
                array.write(writer)?;
            }
        }
        writeln!(writer, "\n  </AppendedData>\n</VTKFile>")
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

pub trait Scalar: Sized + Copy {
    const TYPE_NAME: &'static str;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()>;
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

impl<const D: usize> Scalar for [f32; D] {
    const TYPE_NAME: &'static str = "Float32";

    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        for val in self {
            writer.write_all(&val.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl<const D: usize> Scalar for [f64; D] {
    const TYPE_NAME: &'static str = "Float64";

    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        for val in self {
            writer.write_all(&val.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl<T: Scalar> Scalar for &T {
    const TYPE_NAME: &'static str = T::TYPE_NAME;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        (*self).write_ne_bytes(writer)
    }
}

trait WriteableIter {
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<()>;
}

impl<I, T> WriteableIter for I
where
    I: Iterator<Item = T>,
    T: Scalar,
{
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<()> {
        for item in self {
            item.write_ne_bytes(writer)?;
        }
        Ok(())
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
        Self {
            data_type: "Bit",
            name: name.to_string(),
            number_of_tuples,
            number_of_components,
            data: Box::new(data.into_iter()),
            byte_len: len.div_ceil(8),
        }
    }

    fn new<IT>(name: &str, number_of_components: usize, len: usize, data: IT) -> Self
    where
        IT: IntoIterator + 'a,
        IT::Item: Scalar,
    {
        let number_of_tuples = len / number_of_components;
        Self {
            data_type: <IT::Item as Scalar>::TYPE_NAME,
            name: name.to_string(),
            number_of_tuples,
            number_of_components,
            data: Box::new(data.into_iter()),
            byte_len: len * std::mem::size_of::<IT::Item>(),
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
        self.data.write_to(writer)
    }
}
