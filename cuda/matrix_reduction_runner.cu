#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kDefaultRows = 1000;
constexpr unsigned int kDefaultCols = 1000;
constexpr unsigned int kDefaultWarmup = 3;
constexpr unsigned int kDefaultSamples = 10;
constexpr std::size_t kPreviewColumns = 5;

struct Options {
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

double matrix_value(std::size_t row, std::size_t col) {
  return (static_cast<double>(row) * 0.5) +
         (static_cast<double>(col) * 0.25) +
         (static_cast<double>((row ^ col) & 7U) * 0.125);
}

void fill_matrix(std::vector<double>& values, unsigned int rows, unsigned int cols) {
  for (unsigned int row = 0; row < rows; row += 1) {
    const std::size_t row_start = static_cast<std::size_t>(row) * cols;

    for (unsigned int col = 0; col < cols; col += 1) {
      values[row_start + col] = matrix_value(row, col);
    }
  }
}

double aggregate_columns_and_total(const std::vector<double>& values,
                                   unsigned int rows,
                                   unsigned int cols,
                                   std::vector<double>& column_sums) {
  std::fill(column_sums.begin(), column_sums.end(), 0.0);

  for (unsigned int row = 0; row < rows; row += 1) {
    const std::size_t row_start = static_cast<std::size_t>(row) * cols;

    for (unsigned int col = 0; col < cols; col += 1) {
      column_sums[col] += values[row_start + col];
    }
  }

  return std::accumulate(column_sums.begin(), column_sums.end(), 0.0);
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

    if (arg == "--rows") {
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

__global__ void matrix_column_sum_f64(const double* matrix,
                                      double* column_sums,
                                      unsigned int rows,
                                      unsigned int cols) {
  const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (col < cols) {
    double acc = 0.0;

    for (unsigned int row = 0; row < rows; row += 1) {
      acc += matrix[(static_cast<std::size_t>(row) * cols) + col];
    }

    column_sums[col] = acc;
  }
}

__global__ void matrix_total_sum_f64(const double* column_sums,
                                     double* total,
                                     unsigned int cols) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double acc = 0.0;

    for (unsigned int col = 0; col < cols; col += 1) {
      acc += column_sums[col];
    }

    total[0] = acc;
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

    const std::size_t cells = static_cast<std::size_t>(options.rows) * options.cols;
    const std::size_t matrix_bytes = cells * sizeof(double);
    const std::size_t column_bytes = static_cast<std::size_t>(options.cols) * sizeof(double);

    std::vector<double> host_matrix(cells);
    std::vector<double> reference_column_sums(options.cols);
    std::vector<double> gpu_column_sums(options.cols);
    double gpu_total = 0.0;

    fill_matrix(host_matrix, options.rows, options.cols);
    const double reference_total = aggregate_columns_and_total(
        host_matrix, options.rows, options.cols, reference_column_sums);

    double* device_matrix = nullptr;
    double* device_column_sums = nullptr;
    double* device_total = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_matrix), matrix_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_column_sums), column_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_total), sizeof(double)));

    const dim3 block(256);
    const dim3 grid((options.cols + block.x - 1) / block.x);

    auto run_once = [&]() -> double {
      const auto started_at = std::chrono::steady_clock::now();

      CHECK_CUDA(cudaMemcpy(
          device_matrix, host_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice));

      matrix_column_sum_f64<<<grid, block>>>(
          device_matrix, device_column_sums, options.rows, options.cols);
      CHECK_CUDA(cudaGetLastError());

      matrix_total_sum_f64<<<1, 1>>>(device_column_sums, device_total, options.cols);
      CHECK_CUDA(cudaGetLastError());

      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaMemcpy(
          gpu_column_sums.data(), device_column_sums, column_bytes, cudaMemcpyDeviceToHost));
      CHECK_CUDA(
          cudaMemcpy(&gpu_total, device_total, sizeof(double), cudaMemcpyDeviceToHost));

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

      if (gpu_total != reference_total) {
        std::cerr << "GPU total mismatch: " << gpu_total << " != " << reference_total
                  << std::endl;
        return 3;
      }

      if (gpu_column_sums != reference_column_sums) {
        std::cerr << "GPU column sums mismatch" << std::endl;
        return 4;
      }
    }

    const Stats stats = compute_stats(samples_ms);
    const double touched_bytes = static_cast<double>(matrix_bytes + (2 * column_bytes));
    const double gib_per_second =
        touched_bytes / (1024.0 * 1024.0 * 1024.0) / (stats.average_ms / 1000.0);

    std::cout << "{";
    std::cout << "\"name\":\"gpu / cuda runtime (h2d + d2h)\",";
    std::cout << "\"deviceName\":\"" << escape_json(device.name) << "\",";
    std::cout << "\"averageMs\":" << std::setprecision(17) << stats.average_ms << ",";
    std::cout << "\"medianMs\":" << std::setprecision(17) << stats.median_ms << ",";
    std::cout << "\"minMs\":" << std::setprecision(17) << stats.min_ms << ",";
    std::cout << "\"maxMs\":" << std::setprecision(17) << stats.max_ms << ",";
    std::cout << "\"gibPerSecond\":" << std::setprecision(17) << gib_per_second << ",";
    std::cout << "\"total\":" << std::setprecision(17) << gpu_total << ",";
    std::cout << "\"columnPreview\":" << format_preview(gpu_column_sums);
    std::cout << "}" << std::endl;

    CHECK_CUDA(cudaFree(device_total));
    CHECK_CUDA(cudaFree(device_column_sums));
    CHECK_CUDA(cudaFree(device_matrix));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
