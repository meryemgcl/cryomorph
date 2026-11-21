use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrcHeader {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub mode: i32,
    pub cella: [f32; 3],
    pub cellb: [f32; 3],
}

pub struct MrcReader;

impl MrcReader {
    pub fn read_header<P: AsRef<Path>>(path: P) -> Result<MrcHeader> {
        let mut file = File::open(&path)
            .with_context(|| format!("Failed to open MRC file: {}", path.as_ref().display()))?;

        let nx = file.read_i32::<LittleEndian>()?;
        let ny = file.read_i32::<LittleEndian>()?;
        let nz = file.read_i32::<LittleEndian>()?;
        let mode = file.read_i32::<LittleEndian>()?;

        file.seek(SeekFrom::Start(40))?;
        let mut cella = [0.0f32; 3];
        for val in &mut cella {
            *val = file.read_f32::<LittleEndian>()?;
        }

        let mut cellb = [0.0f32; 3];
        for val in &mut cellb {
            *val = file.read_f32::<LittleEndian>()?;
        }

        Ok(MrcHeader {
            nx,
            ny,
            nz,
            mode,
            cella,
            cellb,
        })
    }
}

/// Zero-copy C-ABI export for Julia and Python FFI
#[no_mangle]
pub extern "C" fn compute_ctf_envelope(
    freq_sq: f64,
    defocus: f64,
    cs: f64,
    wavelength: f64,
    b_factor: f64,
) -> f64 {
    let pi = std::f64::consts::PI;
    let phase_shift = pi * wavelength * freq_sq * defocus - 0.5 * pi * cs * wavelength.powi(3) * freq_sq.powi(2);
    let envelope = (-b_factor * freq_sq / 4.0).exp();
    -phase_shift.sin() * envelope
}
