use std::mem;

use rayon::prelude::*;

pub const F64_BYTES: usize = mem::size_of::<f64>();

pub fn matrix_value(matrix: usize, row: usize, col: usize) -> f64 {
    (matrix as f64 * 0.75)
        + (row as f64 * 0.5)
        + (col as f64 * 0.25)
        + (((matrix ^ row ^ col) & 7) as f64 * 0.125)
}

pub fn checked_cells(rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("Matrix dimensions overflow usize: {rows}x{cols}"))
}

pub fn checked_batch_cells(matrices: usize, rows: usize, cols: usize) -> Result<usize, String> {
    let cells_per_matrix = checked_cells(rows, cols)?;

    cells_per_matrix.checked_mul(matrices).ok_or_else(|| {
        format!("Matrix batch dimensions overflow usize: {matrices} x {rows} x {cols}")
    })
}

pub fn fill_matrices(
    values: &mut [f64],
    matrices: usize,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    ensure_non_zero_shape(matrices, rows, cols)?;

    let expected_cells = checked_batch_cells(matrices, rows, cols)?;

    if values.len() != expected_cells {
        return Err(format!(
            "Expected exactly {expected_cells} f64 values for {matrices} matrices of shape {rows}x{cols}, got {}",
            values.len()
        ));
    }

    let cells_per_matrix = checked_cells(rows, cols)?;

    for matrix in 0..matrices {
        let matrix_offset = matrix * cells_per_matrix;

        for row in 0..rows {
            let row_start = matrix_offset + (row * cols);

            for col in 0..cols {
                values[row_start + col] = matrix_value(matrix, row, col);
            }
        }
    }

    Ok(())
}

pub fn aggregate_matrix_batch(
    values: &[f64],
    matrices: usize,
    rows: usize,
    cols: usize,
    average_column_sums: &mut [f64],
) -> Result<f64, String> {
    ensure_non_zero_shape(matrices, rows, cols)?;

    let expected_cells = checked_batch_cells(matrices, rows, cols)?;

    if values.len() != expected_cells {
        return Err(format!(
            "Expected exactly {expected_cells} f64 values for {matrices} matrices of shape {rows}x{cols}, got {}",
            values.len()
        ));
    }

    if average_column_sums.len() != cols {
        return Err(format!(
            "Expected exactly {cols} averaged column sums, got {}",
            average_column_sums.len()
        ));
    }

    average_column_sums.fill(0.0);

    let cells_per_matrix = checked_cells(rows, cols)?;
    let mut matrix_column_sums = vec![0.0; cols];
    let mut grand_total = 0.0;

    for matrix in 0..matrices {
        matrix_column_sums.fill(0.0);
        let matrix_offset = matrix * cells_per_matrix;

        for row in 0..rows {
            let row_start = matrix_offset + (row * cols);

            for col in 0..cols {
                matrix_column_sums[col] += values[row_start + col];
            }
        }

        let matrix_total = matrix_column_sums.iter().copied().sum::<f64>();

        for col in 0..cols {
            average_column_sums[col] += matrix_column_sums[col];
        }

        grand_total += matrix_total;
    }

    let divisor = matrices as f64;
    for value in average_column_sums.iter_mut() {
        *value /= divisor;
    }

    Ok(grand_total)
}

pub fn aggregate_matrix_batch_parallel(
    values: &[f64],
    matrices: usize,
    rows: usize,
    cols: usize,
    average_column_sums: &mut [f64],
) -> Result<f64, String> {
    ensure_non_zero_shape(matrices, rows, cols)?;

    let expected_cells = checked_batch_cells(matrices, rows, cols)?;

    if values.len() != expected_cells {
        return Err(format!(
            "Expected exactly {expected_cells} f64 values for {matrices} matrices of shape {rows}x{cols}, got {}",
            values.len()
        ));
    }

    if average_column_sums.len() != cols {
        return Err(format!(
            "Expected exactly {cols} averaged column sums, got {}",
            average_column_sums.len()
        ));
    }

    average_column_sums.fill(0.0);

    let cells_per_matrix = checked_cells(rows, cols)?;
    let per_matrix = (0..matrices)
        .into_par_iter()
        .map(|matrix| {
            let matrix_offset = matrix * cells_per_matrix;
            let mut matrix_column_sums = vec![0.0; cols];

            for row in 0..rows {
                let row_start = matrix_offset + (row * cols);

                for col in 0..cols {
                    matrix_column_sums[col] += values[row_start + col];
                }
            }

            let matrix_total = matrix_column_sums.iter().copied().sum::<f64>();
            (matrix_column_sums, matrix_total)
        })
        .collect::<Vec<_>>();

    let mut grand_total = 0.0;
    for (matrix_column_sums, matrix_total) in per_matrix {
        for col in 0..cols {
            average_column_sums[col] += matrix_column_sums[col];
        }

        grand_total += matrix_total;
    }

    let divisor = matrices as f64;
    for value in average_column_sums.iter_mut() {
        *value /= divisor;
    }

    Ok(grand_total)
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

fn ensure_non_zero_shape(matrices: usize, rows: usize, cols: usize) -> Result<(), String> {
    if matrices == 0 || rows == 0 || cols == 0 {
        return Err(format!(
            "Expected non-zero matrix batch dimensions, got matrices={matrices}, rows={rows}, cols={cols}"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        aggregate_matrix_batch, aggregate_matrix_batch_parallel, fill_matrices, matrix_value,
    };

    #[test]
    fn matrix_formula_is_stable() {
        assert_eq!(matrix_value(0, 0, 0), 0.0);
        assert_eq!(matrix_value(1, 1, 2), 2.0);
        assert_eq!(matrix_value(3, 999, 999), 751.875);
    }

    #[test]
    fn fills_and_aggregates_a_small_batch() {
        let matrices = 2usize;
        let rows = 2usize;
        let cols = 3usize;
        let mut values = vec![0.0; matrices * rows * cols];
        let mut average_column_sums = vec![0.0; cols];

        fill_matrices(&mut values, matrices, rows, cols).expect("fill should succeed");
        let grand_total =
            aggregate_matrix_batch(&values, matrices, rows, cols, &mut average_column_sums)
                .expect("aggregation should succeed");

        assert_eq!(average_column_sums, vec![1.375, 1.875, 2.875]);
        assert_eq!(grand_total, 12.25);
    }

    #[test]
    fn parallel_aggregation_matches_the_sequential_path() {
        let matrices = 4usize;
        let rows = 32usize;
        let cols = 16usize;
        let mut values = vec![0.0; matrices * rows * cols];
        let mut sequential_average_column_sums = vec![0.0; cols];
        let mut parallel_average_column_sums = vec![0.0; cols];

        fill_matrices(&mut values, matrices, rows, cols).expect("fill should succeed");
        let sequential_grand_total = aggregate_matrix_batch(
            &values,
            matrices,
            rows,
            cols,
            &mut sequential_average_column_sums,
        )
        .expect("sequential aggregation should succeed");
        let parallel_grand_total = aggregate_matrix_batch_parallel(
            &values,
            matrices,
            rows,
            cols,
            &mut parallel_average_column_sums,
        )
        .expect("parallel aggregation should succeed");

        assert_eq!(parallel_grand_total, sequential_grand_total);
        assert_eq!(parallel_average_column_sums, sequential_average_column_sums);
    }
}
