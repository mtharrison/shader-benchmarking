#include <cublas_v2.h>
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

struct E2ESampleStats {
  double total_ms = 0.0;
  double host_to_device_copy_ms = 0.0;
};

void check_cuda(cudaError_t status, const char* expr);

template <typename T>
struct PinnedHostBuffer {
  T* data = nullptr;
  std::size_t count = 0;

  explicit PinnedHostBuffer(std::size_t requested_count) : count(requested_count) {
    if (count == 0) {
      return;
    }

    void* raw = nullptr;
    check_cuda(cudaMallocHost(&raw, count * sizeof(T)), "cudaMallocHost(&raw, count * sizeof(T))");
    data = static_cast<T*>(raw);
  }

  ~PinnedHostBuffer() {
    if (data != nullptr) {
      cudaFreeHost(data);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  T& operator[](std::size_t index) { return data[index]; }
  const T& operator[](std::size_t index) const { return data[index]; }
};

void check_cuda(cudaError_t status, const char* expr) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(status) +
        " while executing `" + expr + "`");
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)

const char* cublas_status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

void check_cublas(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error: ") + cublas_status_name(status) +
                             " while executing `" + expr + "`");
  }
}

#define CHECK_CUBLAS(expr) check_cublas((expr), #expr)

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

std::string format_result_json(const char* mode,
                               const char* name,
                               const char* device_name,
                               const Stats& stats,
                               double gib_per_second,
                               double grand_total,
                               const std::vector<double>& average_column_sums,
                               double host_to_device_copy_average_ms = -1.0) {
  std::ostringstream output;
  output << "{";
  output << "\"mode\":\"" << escape_json(mode) << "\",";
  output << "\"name\":\"" << escape_json(name) << "\",";
  output << "\"deviceName\":\"" << escape_json(device_name) << "\",";
  output << "\"averageMs\":" << std::setprecision(17) << stats.average_ms << ",";
  output << "\"medianMs\":" << std::setprecision(17) << stats.median_ms << ",";
  output << "\"minMs\":" << std::setprecision(17) << stats.min_ms << ",";
  output << "\"maxMs\":" << std::setprecision(17) << stats.max_ms << ",";
  output << "\"gibPerSecond\":" << std::setprecision(17) << gib_per_second << ",";
  output << "\"grandTotal\":" << std::setprecision(17) << grand_total << ",";
  if (host_to_device_copy_average_ms >= 0.0) {
    output << "\"hostToDeviceCopyAverageMs\":" << std::setprecision(17)
           << host_to_device_copy_average_ms << ",";
  } else {
    output << "\"hostToDeviceCopyAverageMs\":null,";
  }
  output << "\"averageColumnPreview\":" << format_preview(average_column_sums);
  output << "}";
  return output.str();
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
    std::vector<double> host_ones_rows(options.rows, 1.0);
    std::vector<double> host_ones_cols(options.cols, 1.0);
    std::vector<double> host_ones_matrices(options.matrices, 1.0);
    PinnedHostBuffer<double> pinned_host_matrices(cells);
    PinnedHostBuffer<double> pinned_average_column_sums(options.cols);
    PinnedHostBuffer<double> pinned_grand_total(1);
    std::vector<double> reference_average_column_sums(options.cols);
    std::vector<double> resident_average_column_sums(options.cols);
    std::vector<double> pageable_e2e_average_column_sums(options.cols);
    std::vector<double> pinned_e2e_average_column_sums(options.cols);
    double resident_grand_total = 0.0;
    double pageable_e2e_grand_total = 0.0;
    double pinned_e2e_grand_total = 0.0;

    fill_matrices(host_matrices, options.matrices, options.rows, options.cols);
    std::copy(host_matrices.begin(), host_matrices.end(), pinned_host_matrices.data);
    const double reference_grand_total = aggregate_average_columns_and_grand_total(
        host_matrices,
        options.matrices,
        options.rows,
        options.cols,
        reference_average_column_sums);

    double* device_matrices = nullptr;
    double* device_ones_rows = nullptr;
    double* device_ones_cols = nullptr;
    double* device_ones_matrices = nullptr;
    double* device_matrix_column_sums = nullptr;
    double* device_average_column_sums = nullptr;
    double* device_matrix_totals = nullptr;
    double* device_grand_total = nullptr;
    double* device_alpha_one = nullptr;
    double* device_beta_zero = nullptr;
    double* device_alpha_average = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_matrices), matrix_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_ones_rows),
                          static_cast<std::size_t>(options.rows) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_ones_cols),
                          static_cast<std::size_t>(options.cols) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_ones_matrices),
                          static_cast<std::size_t>(options.matrices) * sizeof(double)));
    CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void**>(&device_matrix_column_sums), matrix_column_sum_bytes));
    CHECK_CUDA(
        cudaMalloc(reinterpret_cast<void**>(&device_average_column_sums), average_column_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_matrix_totals), matrix_total_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_grand_total), sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_alpha_one), sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_beta_zero), sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_alpha_average), sizeof(double)));
    CHECK_CUDA(cudaMemcpy(
        device_matrices, host_matrices.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_ones_rows,
                          host_ones_rows.data(),
                          static_cast<std::size_t>(options.rows) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_ones_cols,
                          host_ones_cols.data(),
                          static_cast<std::size_t>(options.cols) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_ones_matrices,
                          host_ones_matrices.data(),
                          static_cast<std::size_t>(options.matrices) * sizeof(double),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));

    const double alpha_one = 1.0;
    const double beta_zero = 0.0;
    const double alpha_average = 1.0 / static_cast<double>(options.matrices);
    CHECK_CUDA(cudaMemcpy(device_alpha_one, &alpha_one, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_beta_zero, &beta_zero, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(device_alpha_average, &alpha_average, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    auto run_reduction_pipeline = [&]() {
      CHECK_CUBLAS(cublasDgemmStridedBatched(handle,
                                             CUBLAS_OP_N,
                                             CUBLAS_OP_N,
                                             static_cast<int>(options.cols),
                                             1,
                                             static_cast<int>(options.rows),
                                             device_alpha_one,
                                             device_matrices,
                                             static_cast<int>(options.cols),
                                             static_cast<long long>(cells_per_matrix),
                                             device_ones_rows,
                                             static_cast<int>(options.rows),
                                             0,
                                             device_beta_zero,
                                             device_matrix_column_sums,
                                             static_cast<int>(options.cols),
                                             static_cast<long long>(options.cols),
                                             static_cast<int>(options.matrices)));

      CHECK_CUBLAS(cublasDgemv(handle,
                               CUBLAS_OP_N,
                               static_cast<int>(options.cols),
                               static_cast<int>(options.matrices),
                               device_alpha_average,
                               device_matrix_column_sums,
                               static_cast<int>(options.cols),
                               device_ones_matrices,
                               1,
                               device_beta_zero,
                               device_average_column_sums,
                               1));

      CHECK_CUBLAS(cublasDgemv(handle,
                               CUBLAS_OP_T,
                               static_cast<int>(options.cols),
                               static_cast<int>(options.matrices),
                               device_alpha_one,
                               device_matrix_column_sums,
                               static_cast<int>(options.cols),
                               device_ones_cols,
                               1,
                               device_beta_zero,
                               device_matrix_totals,
                               1));

      CHECK_CUBLAS(cublasDdot(handle,
                              static_cast<int>(options.matrices),
                              device_matrix_totals,
                              1,
                              device_ones_matrices,
                              1,
                              device_grand_total));
    };

    auto run_once_resident = [&]() -> double {
      const auto started_at = std::chrono::steady_clock::now();

      run_reduction_pipeline();

      CHECK_CUDA(cudaDeviceSynchronize());

      const auto finished_at = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(finished_at - started_at).count();
    };

    auto run_once_e2e_pageable = [&]() -> E2ESampleStats {
      const auto started_at = std::chrono::steady_clock::now();
      const auto copy_started_at = std::chrono::steady_clock::now();

      CHECK_CUDA(
          cudaMemcpy(device_matrices, host_matrices.data(), matrix_bytes, cudaMemcpyHostToDevice));

      const auto copy_finished_at = std::chrono::steady_clock::now();

      run_reduction_pipeline();

      CHECK_CUDA(cudaMemcpy(pageable_e2e_average_column_sums.data(),
                            device_average_column_sums,
                            average_column_bytes,
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(
          cudaMemcpy(&pageable_e2e_grand_total,
                     device_grand_total,
                     sizeof(double),
                     cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaDeviceSynchronize());

      const auto finished_at = std::chrono::steady_clock::now();
      return {
          std::chrono::duration<double, std::milli>(finished_at - started_at).count(),
          std::chrono::duration<double, std::milli>(copy_finished_at - copy_started_at).count(),
      };
    };

    auto run_once_e2e_pinned = [&]() -> E2ESampleStats {
      const auto started_at = std::chrono::steady_clock::now();
      const auto copy_started_at = std::chrono::steady_clock::now();

      CHECK_CUDA(cudaMemcpy(
          device_matrices, pinned_host_matrices.data, matrix_bytes, cudaMemcpyHostToDevice));

      const auto copy_finished_at = std::chrono::steady_clock::now();

      run_reduction_pipeline();

      CHECK_CUDA(cudaMemcpy(pinned_average_column_sums.data,
                            device_average_column_sums,
                            average_column_bytes,
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(
          pinned_grand_total.data, device_grand_total, sizeof(double), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaDeviceSynchronize());

      const auto finished_at = std::chrono::steady_clock::now();
      return {
          std::chrono::duration<double, std::milli>(finished_at - started_at).count(),
          std::chrono::duration<double, std::milli>(copy_finished_at - copy_started_at).count(),
      };
    };

    for (unsigned int iteration = 0; iteration < options.warmup; iteration += 1) {
      const double ignored = run_once_resident();
      (void)ignored;
    }

    std::vector<double> resident_samples_ms;
    resident_samples_ms.reserve(options.samples);

    for (unsigned int iteration = 0; iteration < options.samples; iteration += 1) {
      resident_samples_ms.push_back(run_once_resident());
    }

    CHECK_CUDA(cudaMemcpy(resident_average_column_sums.data(),
                          device_average_column_sums,
                          average_column_bytes,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        &resident_grand_total, device_grand_total, sizeof(double), cudaMemcpyDeviceToHost));

    if (resident_grand_total != reference_grand_total) {
      std::cerr << "Resident GPU grand total mismatch: " << resident_grand_total << " != "
                << reference_grand_total << std::endl;
      return 3;
    }

    if (resident_average_column_sums != reference_average_column_sums) {
      std::cerr << "Resident GPU averaged column sums mismatch" << std::endl;
      return 4;
    }

    for (unsigned int iteration = 0; iteration < options.warmup; iteration += 1) {
      const E2ESampleStats ignored = run_once_e2e_pageable();
      (void)ignored;
    }

    std::vector<double> pageable_e2e_samples_ms;
    pageable_e2e_samples_ms.reserve(options.samples);
    std::vector<double> pageable_h2d_copy_samples_ms;
    pageable_h2d_copy_samples_ms.reserve(options.samples);

    for (unsigned int iteration = 0; iteration < options.samples; iteration += 1) {
      const E2ESampleStats sample = run_once_e2e_pageable();
      pageable_e2e_samples_ms.push_back(sample.total_ms);
      pageable_h2d_copy_samples_ms.push_back(sample.host_to_device_copy_ms);
    }

    if (pageable_e2e_grand_total != reference_grand_total) {
      std::cerr << "Pageable end-to-end GPU grand total mismatch: " << pageable_e2e_grand_total
                << " != "
                << reference_grand_total << std::endl;
      return 5;
    }

    if (pageable_e2e_average_column_sums != reference_average_column_sums) {
      std::cerr << "Pageable end-to-end GPU averaged column sums mismatch" << std::endl;
      return 6;
    }

    for (unsigned int iteration = 0; iteration < options.warmup; iteration += 1) {
      const E2ESampleStats ignored = run_once_e2e_pinned();
      (void)ignored;
    }

    std::vector<double> pinned_e2e_samples_ms;
    pinned_e2e_samples_ms.reserve(options.samples);
    std::vector<double> pinned_h2d_copy_samples_ms;
    pinned_h2d_copy_samples_ms.reserve(options.samples);

    for (unsigned int iteration = 0; iteration < options.samples; iteration += 1) {
      const E2ESampleStats sample = run_once_e2e_pinned();
      pinned_e2e_samples_ms.push_back(sample.total_ms);
      pinned_h2d_copy_samples_ms.push_back(sample.host_to_device_copy_ms);
    }

    pinned_e2e_grand_total = pinned_grand_total[0];
    std::copy_n(
        pinned_average_column_sums.data, options.cols, pinned_e2e_average_column_sums.begin());

    if (pinned_e2e_grand_total != reference_grand_total) {
      std::cerr << "Pinned end-to-end GPU grand total mismatch: " << pinned_e2e_grand_total
                << " != " << reference_grand_total << std::endl;
      return 7;
    }

    if (pinned_e2e_average_column_sums != reference_average_column_sums) {
      std::cerr << "Pinned end-to-end GPU averaged column sums mismatch" << std::endl;
      return 8;
    }

    const Stats resident_stats = compute_stats(resident_samples_ms);
    const Stats pageable_e2e_stats = compute_stats(pageable_e2e_samples_ms);
    const Stats pageable_h2d_copy_stats = compute_stats(pageable_h2d_copy_samples_ms);
    const Stats pinned_e2e_stats = compute_stats(pinned_e2e_samples_ms);
    const Stats pinned_h2d_copy_stats = compute_stats(pinned_h2d_copy_samples_ms);
    const double logical_touched_bytes =
        static_cast<double>(matrix_bytes + (3 * matrix_column_sum_bytes) +
                            average_column_bytes + (2 * matrix_total_bytes) + sizeof(double));
    const double e2e_touched_bytes =
        logical_touched_bytes + matrix_bytes + average_column_bytes + sizeof(double);
    const double resident_gib_per_second =
        logical_touched_bytes / (1024.0 * 1024.0 * 1024.0) / (resident_stats.average_ms / 1000.0);
    const double pageable_e2e_gib_per_second = e2e_touched_bytes / (1024.0 * 1024.0 * 1024.0) /
                                               (pageable_e2e_stats.average_ms / 1000.0);
    const double pinned_e2e_gib_per_second = e2e_touched_bytes / (1024.0 * 1024.0 * 1024.0) /
                                             (pinned_e2e_stats.average_ms / 1000.0);

    std::cout << "[";
    std::cout << format_result_json("e2e-pageable",
                                    "gpu / cuda e2e pageable batch aggregation (cuBLAS)",
                                    device.name,
                                    pageable_e2e_stats,
                                    pageable_e2e_gib_per_second,
                                    pageable_e2e_grand_total,
                                    pageable_e2e_average_column_sums,
                                    pageable_h2d_copy_stats.average_ms);
    std::cout << ",";
    std::cout << format_result_json("e2e-pinned",
                                    "gpu / cuda e2e pinned batch aggregation (cuBLAS)",
                                    device.name,
                                    pinned_e2e_stats,
                                    pinned_e2e_gib_per_second,
                                    pinned_e2e_grand_total,
                                    pinned_e2e_average_column_sums,
                                    pinned_h2d_copy_stats.average_ms);
    std::cout << ",";
    std::cout << format_result_json("resident",
                                    "gpu / cuda resident batch aggregation (cuBLAS)",
                                    device.name,
                                    resident_stats,
                                    resident_gib_per_second,
                                    resident_grand_total,
                                    resident_average_column_sums);
    std::cout << "]" << std::endl;

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_alpha_average));
    CHECK_CUDA(cudaFree(device_beta_zero));
    CHECK_CUDA(cudaFree(device_alpha_one));
    CHECK_CUDA(cudaFree(device_grand_total));
    CHECK_CUDA(cudaFree(device_matrix_totals));
    CHECK_CUDA(cudaFree(device_average_column_sums));
    CHECK_CUDA(cudaFree(device_matrix_column_sums));
    CHECK_CUDA(cudaFree(device_ones_matrices));
    CHECK_CUDA(cudaFree(device_ones_cols));
    CHECK_CUDA(cudaFree(device_ones_rows));
    CHECK_CUDA(cudaFree(device_matrices));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
