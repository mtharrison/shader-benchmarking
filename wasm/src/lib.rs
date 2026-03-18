use std::slice;

const GOLDEN_GAMMA: u32 = 0x9e37_79b9;
const ROTATE_BITS: u32 = 13;
const MULTIPLIER: u32 = 1_664_525;
const INCREMENT: u32 = 1_013_904_223;

#[no_mangle]
pub extern "C" fn alloc_u32_buffer(cells: u32) -> *mut u32 {
    let cells = cells as usize;
    let mut values = vec![0u32; cells];
    let ptr = values.as_mut_ptr();

    std::mem::forget(values);

    ptr
}

#[no_mangle]
pub extern "C" fn free_u32_buffer(ptr: *mut u32, cells: u32) {
    if ptr.is_null() {
        return;
    }

    let cells = cells as usize;

    // SAFETY: `ptr` must have been returned by `alloc_u32_buffer` for `cells` elements.
    unsafe {
        drop(Vec::from_raw_parts(ptr, cells, cells));
    }
}

#[no_mangle]
pub extern "C" fn mutate_u32_buffer(ptr: *mut u32, cells: u32, passes: u32) -> u32 {
    if ptr.is_null() {
        return 0;
    }

    let cells = cells as usize;

    // SAFETY: `ptr` must point to `cells` initialized `u32` values in wasm linear memory.
    let values = unsafe { slice::from_raw_parts_mut(ptr, cells) };
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
