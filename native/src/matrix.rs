use std::mem;

pub const U32_BYTES: usize = mem::size_of::<u32>();

const GOLDEN_GAMMA: u32 = 0x9e37_79b9;
const ROTATE_BITS: u32 = 13;
const MULTIPLIER: u32 = 1_664_525;
const INCREMENT: u32 = 1_013_904_223;

pub fn fill_sequence(values: &mut [u32], seed: u32) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = seed.wrapping_add(index as u32);
    }
}

pub fn mutate_values(values: &mut [u32], passes: u32) -> u32 {
    let mut checksum = 0u32;

    for pass in 0..passes {
        for (index, value) in values.iter_mut().enumerate() {
            let salt = pass.wrapping_add(index as u32).wrapping_mul(GOLDEN_GAMMA);
            let mixed = value.wrapping_add(salt).rotate_left(ROTATE_BITS);
            let next = mixed.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);

            *value = next;
            checksum = checksum.wrapping_add(next);
        }
    }

    checksum
}

pub fn bytes_to_u32_vec(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.len() % U32_BYTES != 0 {
        return Err(format!(
            "Expected a buffer length divisible by {U32_BYTES}, got {}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(U32_BYTES)
        .map(|chunk| {
            let raw: [u8; U32_BYTES] = chunk
                .try_into()
                .expect("chunk_exact always yields 4-byte chunks");

            u32::from_le_bytes(raw)
        })
        .collect())
}

pub fn u32_vec_to_bytes(mut values: Vec<u32>) -> Vec<u8> {
    let len = values.len();
    let capacity = values.capacity();
    let ptr = values.as_mut_ptr();

    mem::forget(values);

    // SAFETY: `ptr` originated from a `Vec<u32>` allocation with `len` initialized elements.
    unsafe { Vec::from_raw_parts(ptr.cast::<u8>(), len * U32_BYTES, capacity * U32_BYTES) }
}

pub fn bytes_as_u32_slice_mut(bytes: &mut [u8]) -> Result<&mut [u32], String> {
    if bytes.len() % U32_BYTES != 0 {
        return Err(format!(
            "Expected a buffer length divisible by {U32_BYTES}, got {}",
            bytes.len()
        ));
    }

    // SAFETY: `align_to_mut` is safe to call; we reject any misaligned prefix/suffix.
    let (prefix, values, suffix) = unsafe { bytes.align_to_mut::<u32>() };

    if !prefix.is_empty() || !suffix.is_empty() {
        return Err("Expected a 4-byte aligned buffer".to_string());
    }

    Ok(values)
}
