use std::mem;

pub const F64_BYTES: usize = mem::size_of::<f64>();

pub fn matrix_value(row: usize, col: usize) -> f64 {
    (row as f64 * 0.5) + (col as f64 * 0.25) + (((row ^ col) & 7) as f64 * 0.125)
}

pub fn fill_matrix(values: &mut [f64], rows: usize, cols: usize) -> Result<(), String> {
    let expected_cells = checked_cells(rows, cols)?;

    if values.len() != expected_cells {
        return Err(format!(
            "Expected exactly {expected_cells} f64 values for a {rows}x{cols} matrix, got {}",
            values.len()
        ));
    }

    for row in 0..rows {
        let row_start = row * cols;

        for col in 0..cols {
            values[row_start + col] = matrix_value(row, col);
        }
    }

    Ok(())
}

pub fn aggregate_columns_and_total(
    values: &[f64],
    rows: usize,
    cols: usize,
    column_sums: &mut [f64],
) -> Result<f64, String> {
    let expected_cells = checked_cells(rows, cols)?;

    if values.len() != expected_cells {
        return Err(format!(
            "Expected exactly {expected_cells} f64 values for a {rows}x{cols} matrix, got {}",
            values.len()
        ));
    }

    if column_sums.len() != cols {
        return Err(format!(
            "Expected exactly {cols} f64 column sums, got {}",
            column_sums.len()
        ));
    }

    column_sums.fill(0.0);

    for row in 0..rows {
        let row_start = row * cols;

        for col in 0..cols {
            column_sums[col] += values[row_start + col];
        }
    }

    Ok(column_sums.iter().copied().sum())
}

pub fn checked_cells(rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("Matrix dimensions overflow usize: {rows}x{cols}"))
}

pub fn bytes_to_f64_vec(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if bytes.len() % F64_BYTES != 0 {
        return Err(format!(
            "Expected a buffer length divisible by {F64_BYTES}, got {}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(F64_BYTES)
        .map(|chunk| {
            let raw: [u8; F64_BYTES] = chunk
                .try_into()
                .expect("chunk_exact always yields 8-byte chunks");

            f64::from_le_bytes(raw)
        })
        .collect())
}

pub fn f64_vec_to_bytes(mut values: Vec<f64>) -> Vec<u8> {
    let len = values.len();
    let capacity = values.capacity();
    let ptr = values.as_mut_ptr();

    mem::forget(values);

    // SAFETY: `ptr` originated from a `Vec<f64>` allocation with `len` initialized elements.
    unsafe { Vec::from_raw_parts(ptr.cast::<u8>(), len * F64_BYTES, capacity * F64_BYTES) }
}

pub fn bytes_as_f64_slice(bytes: &[u8]) -> Result<&[f64], String> {
    if bytes.len() % F64_BYTES != 0 {
        return Err(format!(
            "Expected a buffer length divisible by {F64_BYTES}, got {}",
            bytes.len()
        ));
    }

    let (prefix, values, suffix) = unsafe { bytes.align_to::<f64>() };

    if !prefix.is_empty() || !suffix.is_empty() {
        return Err("Expected an 8-byte aligned buffer".to_string());
    }

    Ok(values)
}

pub fn bytes_as_f64_slice_mut(bytes: &mut [u8]) -> Result<&mut [f64], String> {
    if bytes.len() % F64_BYTES != 0 {
        return Err(format!(
            "Expected a buffer length divisible by {F64_BYTES}, got {}",
            bytes.len()
        ));
    }

    let (prefix, values, suffix) = unsafe { bytes.align_to_mut::<f64>() };

    if !prefix.is_empty() || !suffix.is_empty() {
        return Err("Expected an 8-byte aligned buffer".to_string());
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{aggregate_columns_and_total, fill_matrix, matrix_value};

    #[test]
    fn matrix_formula_is_stable() {
        assert_eq!(matrix_value(0, 0), 0.0);
        assert_eq!(matrix_value(1, 2), 1.375);
        assert_eq!(matrix_value(999, 999), 749.25);
    }

    #[test]
    fn fills_and_aggregates_a_small_matrix() {
        let rows = 3usize;
        let cols = 4usize;
        let mut values = vec![0.0; rows * cols];
        let mut column_sums = vec![0.0; cols];

        fill_matrix(&mut values, rows, cols).expect("fill should succeed");
        let total = aggregate_columns_and_total(&values, rows, cols, &mut column_sums)
            .expect("aggregation should succeed");

        assert_eq!(column_sums, vec![1.875, 2.75, 3.625, 4.5]);
        assert_eq!(total, 12.75);
    }
}
