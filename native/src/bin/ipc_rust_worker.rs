use std::io::{self, ErrorKind, Read, Write};

#[allow(dead_code)]
#[path = "../matrix.rs"]
mod matrix;

use matrix::{mutate_values, U32_BYTES};

const REQUEST_HEADER_BYTES: usize = 8;
const RESPONSE_HEADER_BYTES: usize = 8;

fn main() -> io::Result<()> {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    loop {
        let mut header = [0u8; REQUEST_HEADER_BYTES];

        match stdin.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }

        let passes = u32::from_le_bytes(header[..4].try_into().expect("header is 8 bytes"));
        let byte_length =
            u32::from_le_bytes(header[4..].try_into().expect("header is 8 bytes")) as usize;

        if byte_length % U32_BYTES != 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Expected payload length divisible by {U32_BYTES}, got {byte_length}"),
            ));
        }

        let cells = byte_length / U32_BYTES;
        let mut values = vec![0u32; cells];
        let payload = unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_length)
        };
        stdin.read_exact(payload)?;

        let checksum = mutate_values(&mut values, passes);
        let payload =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_length) };

        let mut response_header = [0u8; RESPONSE_HEADER_BYTES];
        response_header[..4].copy_from_slice(&checksum.to_le_bytes());
        response_header[4..].copy_from_slice(&(byte_length as u32).to_le_bytes());

        stdout.write_all(&response_header)?;
        stdout.write_all(&payload)?;
        stdout.flush()?;
    }

    Ok(())
}
