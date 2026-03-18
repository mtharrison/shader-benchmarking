#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kDefaultMatrices = 4;
constexpr unsigned int kDefaultRows = 1000;
constexpr unsigned int kDefaultCols = 1000;
constexpr unsigned int kDefaultWarmup = 3;
constexpr unsigned int kDefaultSamples = 10;
constexpr unsigned int kReductionBlockSize = 256;
constexpr std::size_t kPreviewColumns = 5;

struct Options {
  unsigned int matrices = kDefaultMatrices;
  unsigned int rows = kDefaultRows;
  unsigned int cols = kDefaultCols;
  unsigned int warmup = kDefaultWarmup;
  unsigned int samples = kDefaultSamples;
};

struct Stats {
  double average_ms = 0.0;
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

void check_cuda(cudaError_t status, const char* expr) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(status) +
        " while executing `" + expr + "`");
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)

double matrix_value(std::size_t matrix, std::size_t row, std::size_t col) {
  return (static_cast<double>(matrix) * 0.75) +
         (static_cast<double>(row) * 0.5) +
         (static_cast<double>(col) * 0.25) +
         (static_cast<double>((matrix ^ row ^ col) & 7U) * 0.125);
}

std::size_t checked_cells_per_matrix(unsigned int rows, unsigned int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t checked_batch_cells(unsigned int matrices, unsigned int rows, unsigned int cols) {
  return static_cast<std::size_t>(matrices) * checked_cells_per_matrix(rows, cols);
}

void fill_matrices(std::vector<double>& values,
                   unsigned int matrices,
                   unsigned int rows,
                   unsigned int cols) {
  const std::size_t cells_per_matrix = checked_cells_per_matrix(rows, cols);

  for (unsigned int matrix = 0; matrix < matrices; matrix += 1) {
    const std::size_t matrix_offset = static_cast<std::size_t>(matrix) * cells_per_matrix;

    for (unsigned int row = 0; row < rows; row += 1) {
      const std::size_t row_start = matrix_offset + (static_cast<std::size_t>(row) * cols);

      for (unsigned int col = 0; col < cols; col += 1) {
        values[row_start + col] = matrix_value(matrix, row, col);
      }
    }
  }
}

double aggregate_average_columns_and_grand_total(const std::vector<double>& values,
                                                 unsigned int matrices,
                                                 unsigned int rows,
                                                 unsigned int cols,
                                                 std::vector<double>& average_column_sums) {
  average_column_sums.assign(cols, 0.0);

  const std::size_t cells_per_matrix = checked_cells_per_matrix(rows, cols);
  std::vector<double> matrix_column_sums(cols, 0.0);
  double grand_total = 0.0;

  for (unsigned int matrix = 0; matrix < matrices; matrix += 1) {
    std::fill(matrix_column_sums.begin(), matrix_column_sums.end(), 0.0);
    const std::size_t matrix_offset = static_cast<std::size_t>(matrix) * cells_per_matrix;

    for (unsigned int row = 0; row < rows; row += 1) {
      const std::size_t row_start = matrix_offset + (static_cast<std::size_t>(row) * cols);

      for (unsigned int col = 0; col < cols; col += 1) {
        matrix_column_sums[col] += values[row_start + col];
      }
    }

    double matrix_total = 0.0;
    for (unsigned int col = 0; col < cols; col += 1) {
      average_column_sums[col] += matrix_column_sums[col];
      matrix_total += matrix_column_sums[col];
    }

    grand_total += matrix_total;
  }

  const double divisor = static_cast<double>(matrices);
  for (double& value : average_column_sums) {
    value /= divisor;
  }

  return grand_total;
}

Stats compute_stats(const std::vector<double>& samples_ms) {
  Stats stats{};
  std::vector<double> sorted = samples_ms;
  std::sort(sorted.begin(), sorted.end());

  stats.average_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
                     static_cast<double>(samples_ms.size());
  stats.median_ms = sorted[sorted.size() / 2];
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();

  return stats;
}

Options parse_args(int argc, char** argv) {
  Options options{};

  for (int index = 1; index < argc; index += 1) {
    const std::string arg = argv[index];

    auto read_value = [&](const char* flag, bool allow_zero) -> unsigned int {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }

      const char* raw = argv[index + 1];
      char* end = nullptr;
      const unsigned long parsed = std::strtoul(raw, &end, 10);

      if (end == raw || *end != '\0' || parsed > 0xffffffffUL ||
          (!allow_zero && parsed == 0)) {
        throw std::runtime_error(std::string("Invalid numeric value for ") + flag + ": " + raw);
      }

      index += 1;
      return static_cast<unsigned int>(parsed);
    };

    if (arg == "--matrices") {
      options.matrices = read_value("--matrices", false);
    } else if (arg == "--rows") {
      options.rows = read_value("--rows", false);
    } else if (arg == "--cols") {
      options.cols = read_value("--cols", false);
    } else if (arg == "--warmup") {
      options.warmup = read_value("--warmup", true);
    } else if (arg == "--samples") {
      options.samples = read_value("--samples", false);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return options;
}

std::string escape_json(const std::string& input) {
  std::ostringstream escaped;

  for (const char ch : input) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      default:
        escaped << ch;
        break;
    }
  }

  return escaped.str();
}

std::string format_preview(const std::vector<double>& values) {
  std::ostringstream preview;
  preview << "[";

  const std::size_t count = std::min<std::size_t>(values.size(), kPreviewColumns);
  for (std::size_t index = 0; index < count; index += 1) {
    if (index > 0) {
      preview << ",";
    }

    preview << std::setprecision(17) << values[index];
  }

  preview << "]";
  return preview.str();
}

__device__ double block_reduce_sum(double partial, double* partials) {
  partials[threadIdx.x] = partial;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }

    __syncthreads();
  }

  return partials[0];
}

__global__ void matrix_column_sum_f64(const double* matrices,
                                      double* matrix_column_sums,
                                      unsigned int matrix_count,
                                      unsigned int rows,
                                      unsigned int cols) {
  const unsigned int linear = blockIdx.x;
  const unsigned int total_columns = matrix_count * cols;

  if (linear < total_columns) {
    const unsigned int matrix = linear / cols;
    const unsigned int col = linear % cols;
    const std::size_t cells_per_matrix = static_cast<std::size_t>(rows) * cols;
    const std::size_t matrix_offset = static_cast<std::size_t>(matrix) * cells_per_matrix;
    extern __shared__ double partials[];
    double partial = 0.0;

    for (unsigned int row = threadIdx.x; row < rows; row += blockDim.x) {
      partial += matrices[matrix_offset + (static_cast<std::size_t>(row) * cols) + col];
    }

    const double reduced = block_reduce_sum(partial, partials);
    if (threadIdx.x == 0) {
      matrix_column_sums[linear] = reduced;
    }
  }
}

__global__ void average_columns_f64(const double* matrix_column_sums,
                                    double* average_column_sums,
                                    unsigned int matrix_count,
                                    unsigned int cols) {
  const unsigned int col = blockIdx.x;

  if (col < cols) {
    extern __shared__ double partials[];
    double partial = 0.0;

    for (unsigned int matrix = threadIdx.x; matrix < matrix_count; matrix += blockDim.x) {
      partial += matrix_column_sums[(static_cast<std::size_t>(matrix) * cols) + col];
    }

    const double reduced = block_reduce_sum(partial, partials);
    if (threadIdx.x == 0) {
      average_column_sums[col] = reduced / static_cast<double>(matrix_count);
    }
  }
}

__global__ void matrix_totals_f64(const double* matrix_column_sums,
                                  double* matrix_totals,
                                  unsigned int matrix_count,
                                  unsigned int cols) {
  const unsigned int matrix = blockIdx.x;

  if (matrix < matrix_count) {
    extern __shared__ double partials[];
    double partial = 0.0;
    const std::size_t row_offset = static_cast<std::size_t>(matrix) * cols;

    for (unsigned int col = threadIdx.x; col < cols; col += blockDim.x) {
      partial += matrix_column_sums[row_offset + col];
    }

    const double reduced = block_reduce_sum(partial, partials);
    if (threadIdx.x == 0) {
      matrix_totals[matrix] = reduced;
    }
  }
}

__global__ void grand_total_f64(const double* matrix_totals,
                                double* grand_total,
                                unsigned int matrix_count) {
  extern __shared__ double partials[];
  double partial = 0.0;

  for (unsigned int matrix = threadIdx.x; matrix < matrix_count; matrix += blockDim.x) {
    partial += matrix_totals[matrix];
  }

  const double reduced = block_reduce_sum(partial, partials);
  if (threadIdx.x == 0) {
    grand_total[0] = reduced;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);

    int device_count = 0;
    const cudaError_t query_status = cudaGetDeviceCount(&device_count);
    if (query_status != cudaSuccess || device_count <= 0) {
      std::cerr << "No NVIDIA GPU available" << std::endl;
      return 2;
    }

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp device{};
    CHECK_CUDA(cudaGetDeviceProperties(&device, 0));

    const std::size_t cells = checked_batch_cells(options.matrices, options.rows, options.cols);
    const std::size_t cells_per_matrix = checked_cells_per_matrix(options.rows, options.cols);
    const std::size_t matrix_bytes = cells * sizeof(double);
    const std::size_t matrix_column_sum_bytes =
        static_cast<std::size_t>(options.matrices) * options.cols * sizeof(double);
    const std::size_t average_column_bytes =
        static_cast<std::size_t>(options.cols) * sizeof(double);
    const std::size_t matrix_total_bytes =
        static_cast<std::size_t>(options.matrices) * sizeof(double);

    std::vector<double> host_matrices(cells);
    std::vector<double> reference_average_column_sums(options.cols);
    std::vector<double> gpu_average_column_sums(options.cols);
    double gpu_grand_total = 0.0;

    fill_matrices(host_matrices, options.matrices, options.rows, options.cols);
    const double reference_grand_total = aggregate_average_columns_and_grand_total(
        host_matrices,
        options.matrices,
        options.rows,
        options.cols,
        reference_average_column_sums);

    double* device_matrices = nullptr;
    double* device_matrix_column_sums = nullptr;
    double* device_average_column_sums = nullptr;
    double* device_matrix_totals = nullptr;
    double* device_grand_total = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_matrices), matrix_bytes));
    CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void**>(&device_matrix_column_sums), matrix_column_sum_bytes));
    CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void**>(&device_average_column_sums), average_column_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_matrix_totals), matrix_total_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_grand_total), sizeof(double)));

    const dim3 block(kReductionBlockSize);
    const std::size_t reduction_shared_bytes = static_cast<std::size_t>(block.x) * sizeof(double);
    const dim3 column_sum_grid(static_cast<unsigned int>(
        static_cast<std::size_t>(options.matrices) * options.cols));
    const dim3 average_columns_grid(options.cols);
    const dim3 matrix_totals_grid(options.matrices);

    auto run_once = [&]() -> double {
      const auto started_at = std::chrono::steady_clock::now();

      CHECK_CUDA(cudaMemcpy(
          device_matrices, host_matrices.data(), matrix_bytes, cudaMemcpyHostToDevice));

      matrix_column_sum_f64<<<column_sum_grid, block, reduction_shared_bytes>>>(
          device_matrices,
          device_matrix_column_sums,
          options.matrices,
          options.rows,
          options.cols);
      CHECK_CUDA(cudaGetLastError());

      average_columns_f64<<<average_columns_grid, block, reduction_shared_bytes>>>(
          device_matrix_column_sums,
          device_average_column_sums,
          options.matrices,
          options.cols);
      CHECK_CUDA(cudaGetLastError());

      matrix_totals_f64<<<matrix_totals_grid, block, reduction_shared_bytes>>>(
          device_matrix_column_sums,
          device_matrix_totals,
          options.matrices,
          options.cols);
      CHECK_CUDA(cudaGetLastError());

      grand_total_f64<<<1, block, reduction_shared_bytes>>>(
          device_matrix_totals, device_grand_total, options.matrices);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(
          gpu_average_column_sums.data(),
          device_average_column_sums,
          average_column_bytes,
          cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(
          &gpu_grand_total,
          device_grand_total,
          sizeof(double),
          cudaMemcpyDeviceToHost));

      const auto finished_at = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(finished_at - started_at).count();
    };

    for (unsigned int iteration = 0; iteration < options.warmup; iteration += 1) {
      const double ignored = run_once();
      (void)ignored;
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(options.samples);

    for (unsigned int iteration = 0; iteration < options.samples; iteration += 1) {
      samples_ms.push_back(run_once());

      if (gpu_grand_total != reference_grand_total) {
        std::cerr << "GPU grand total mismatch: " << gpu_grand_total << " != "
                  << reference_grand_total << std::endl;
        return 3;
      }

      if (gpu_average_column_sums != reference_average_column_sums) {
        std::cerr << "GPU averaged column sums mismatch" << std::endl;
        return 4;
      }
    }

    const Stats stats = compute_stats(samples_ms);
    const double touched_bytes = static_cast<double>(matrix_bytes + average_column_bytes +
                                                     sizeof(double));
    const double gib_per_second =
        touched_bytes / (1024.0 * 1024.0 * 1024.0) / (stats.average_ms / 1000.0);

    std::cout << "{";
    std::cout << "\"name\":\"gpu / cuda batch aggregation (h2d + d2h)\",";
    std::cout << "\"deviceName\":\"" << escape_json(device.name) << "\",";
    std::cout << "\"averageMs\":" << std::setprecision(17) << stats.average_ms << ",";
    std::cout << "\"medianMs\":" << std::setprecision(17) << stats.median_ms << ",";
    std::cout << "\"minMs\":" << std::setprecision(17) << stats.min_ms << ",";
    std::cout << "\"maxMs\":" << std::setprecision(17) << stats.max_ms << ",";
    std::cout << "\"gibPerSecond\":" << std::setprecision(17) << gib_per_second << ",";
    std::cout << "\"grandTotal\":" << std::setprecision(17) << gpu_grand_total << ",";
    std::cout << "\"averageColumnPreview\":" << format_preview(gpu_average_column_sums);
    std::cout << "}" << std::endl;

    CHECK_CUDA(cudaFree(device_grand_total));
    CHECK_CUDA(cudaFree(device_matrix_totals));
    CHECK_CUDA(cudaFree(device_average_column_sums));
    CHECK_CUDA(cudaFree(device_matrix_column_sums));
    CHECK_CUDA(cudaFree(device_matrices));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
