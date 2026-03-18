pub mod f64_matrix;
pub mod gpu_pipeline;
pub mod matrix;

use napi::bindgen_prelude::Buffer;
use napi::{Error, Result};
use napi_derive::napi;

use crate::f64_matrix::{
    aggregate_columns_and_total, bytes_as_f64_slice, bytes_as_f64_slice_mut, checked_cells,
    f64_vec_to_bytes, fill_matrix, matrix_value,
};
use crate::gpu_pipeline::{
    compile_gpu_pipeline, compile_matrix_reduction_pipeline, sample_gpu_map2_source,
    sample_matrix_reduction_source, CompiledGpuPipeline, CompiledMatrixReductionPipeline,
};
use crate::matrix::{
    bytes_as_u32_slice_mut, bytes_to_u32_vec, fill_sequence, mutate_values, u32_vec_to_bytes,
    U32_BYTES,
};

const MATRIX_DIMENSION: usize = 6;
const MATRIX_CELLS: usize = MATRIX_DIMENSION * MATRIX_DIMENSION;
const MATRIX_BYTES: usize = MATRIX_CELLS * U32_BYTES;

#[napi(object)]
pub struct MatrixAggregationResult {
    pub column_sums: Buffer,
    pub total: f64,
}

#[napi]
pub fn create_shared_matrix() -> Buffer {
    let mut values = vec![0u32; MATRIX_CELLS];
    fill_sequence(&mut values, 1);
    Buffer::from(u32_vec_to_bytes(values))
}

#[napi]
pub fn create_u32_buffer(cells: u32) -> Result<Buffer> {
    let cells = usize::try_from(cells)
        .map_err(|_| Error::from_reason("Cell count does not fit into usize".to_string()))?;
    let values = vec![0u32; cells];

    Ok(Buffer::from(u32_vec_to_bytes(values)))
}

#[napi]
pub fn create_f64_buffer(cells: u32) -> Result<Buffer> {
    let cells = usize::try_from(cells)
        .map_err(|_| Error::from_reason("Cell count does not fit into usize".to_string()))?;
    let values = vec![0.0f64; cells];

    Ok(Buffer::from(f64_vec_to_bytes(values)))
}

#[napi]
pub fn create_f64_matrix(rows: u32, cols: u32) -> Result<Buffer> {
    let rows = usize::try_from(rows)
        .map_err(|_| Error::from_reason("Row count does not fit into usize".to_string()))?;
    let cols = usize::try_from(cols)
        .map_err(|_| Error::from_reason("Column count does not fit into usize".to_string()))?;
    let cells = checked_cells(rows, cols).map_err(Error::from_reason)?;
    let mut values = vec![0.0f64; cells];
    fill_matrix(&mut values, rows, cols).map_err(Error::from_reason)?;

    Ok(Buffer::from(f64_vec_to_bytes(values)))
}

#[napi]
pub fn mutate_u32_buffer(mut buffer: Buffer, passes: u32) -> Result<u32> {
    let values = bytes_as_u32_slice_mut(buffer.as_mut()).map_err(Error::from_reason)?;
    Ok(mutate_values(values, passes))
}

#[napi(js_name = "printMatrix6x6")]
pub fn print_matrix_6x6(buffer: Buffer) -> Result<()> {
    let values = bytes_to_u32_vec(buffer.as_ref()).map_err(Error::from_reason)?;

    if values.len() != MATRIX_CELLS {
        return Err(Error::from_reason(format!(
            "Expected exactly {MATRIX_BYTES} bytes for a 6x6 u32 matrix, got {}",
            buffer.len()
        )));
    }

    for row in values.chunks_exact(MATRIX_DIMENSION) {
        let line = row
            .iter()
            .map(|value| format!("{value:>4}"))
            .collect::<Vec<_>>()
            .join(" ");

        println!("{line}");
    }

    Ok(())
}

#[napi]
pub fn fill_f64_matrix_buffer(mut buffer: Buffer, rows: u32, cols: u32) -> Result<()> {
    let rows = usize::try_from(rows)
        .map_err(|_| Error::from_reason("Row count does not fit into usize".to_string()))?;
    let cols = usize::try_from(cols)
        .map_err(|_| Error::from_reason("Column count does not fit into usize".to_string()))?;
    let values = bytes_as_f64_slice_mut(buffer.as_mut()).map_err(Error::from_reason)?;
    fill_matrix(values, rows, cols).map_err(Error::from_reason)
}

#[napi]
pub fn f64_matrix_value(row: u32, col: u32) -> f64 {
    matrix_value(row as usize, col as usize)
}

#[napi]
pub fn aggregate_f64_matrix_columns(
    matrix_buffer: Buffer,
    rows: u32,
    cols: u32,
    mut column_sums_buffer: Buffer,
) -> Result<f64> {
    let rows = usize::try_from(rows)
        .map_err(|_| Error::from_reason("Row count does not fit into usize".to_string()))?;
    let cols = usize::try_from(cols)
        .map_err(|_| Error::from_reason("Column count does not fit into usize".to_string()))?;
    let matrix = bytes_as_f64_slice(matrix_buffer.as_ref()).map_err(Error::from_reason)?;
    let column_sums =
        bytes_as_f64_slice_mut(column_sums_buffer.as_mut()).map_err(Error::from_reason)?;

    aggregate_columns_and_total(matrix, rows, cols, column_sums).map_err(Error::from_reason)
}

#[napi]
pub fn aggregate_f64_matrix_columns_allocating(
    matrix_buffer: Buffer,
    rows: u32,
    cols: u32,
) -> Result<MatrixAggregationResult> {
    let rows = usize::try_from(rows)
        .map_err(|_| Error::from_reason("Row count does not fit into usize".to_string()))?;
    let cols = usize::try_from(cols)
        .map_err(|_| Error::from_reason("Column count does not fit into usize".to_string()))?;
    let matrix = bytes_as_f64_slice(matrix_buffer.as_ref()).map_err(Error::from_reason)?;
    let mut column_sums = vec![0.0f64; cols];
    let total = aggregate_columns_and_total(matrix, rows, cols, &mut column_sums)
        .map_err(Error::from_reason)?;

    Ok(MatrixAggregationResult {
        column_sums: Buffer::from(f64_vec_to_bytes(column_sums)),
        total,
    })
}

#[napi]
pub fn sample_gpu_map2_source_code() -> String {
    sample_gpu_map2_source().to_string()
}

#[napi]
pub fn compile_gpu_map2_pipeline(source: String) -> Result<CompiledGpuPipeline> {
    compile_gpu_pipeline(&source).map_err(Error::from_reason)
}

#[napi]
pub fn compile_sample_gpu_map2() -> Result<CompiledGpuPipeline> {
    compile_gpu_pipeline(sample_gpu_map2_source()).map_err(Error::from_reason)
}

#[napi]
pub fn sample_matrix_reduction_source_code() -> String {
    sample_matrix_reduction_source().to_string()
}

#[napi]
pub fn compile_matrix_reduction_gpu_pipeline(
    rows: u32,
    cols: u32,
) -> Result<CompiledMatrixReductionPipeline> {
    compile_matrix_reduction_pipeline(rows as usize, cols as usize).map_err(Error::from_reason)
}
