#include "chetverikova_e_shell_sort_simple_merge/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "chetverikova_e_shell_sort_simple_merge/common/include/common.hpp"

namespace chetverikova_e_shell_sort_simple_merge {

ChetverikovaEShellSortSimpleMergeALL::ChetverikovaEShellSortSimpleMergeALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput().clear();
  }

  GetOutput().clear();
}

bool ChetverikovaEShellSortSimpleMergeALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return !GetInput().empty();
  }

  return true;
}

bool ChetverikovaEShellSortSimpleMergeALL::PreProcessingImpl() {
  return true;
}

void ChetverikovaEShellSortSimpleMergeALL::ShellSort(std::vector<int> &data) {
  if (data.empty()) {
    return;
  }

  size_t n = data.size();

  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; ++i) {
      int temp = data[i];
      size_t j = i;

      while (j >= gap && data[j - gap] > temp) {
        data[j] = data[j - gap];
        j -= gap;
      }

      data[j] = temp;
    }
  }
}

std::vector<int> ChetverikovaEShellSortSimpleMergeALL::MergeTwoSortedVectors(const std::vector<int> &a,
                                                                             const std::vector<int> &b) {
  std::vector<int> result(a.size() + b.size());

  std::ranges::merge(a, b, result.begin());

  return result;
}

void ChetverikovaEShellSortSimpleMergeALL::CalculateCountsAndDisplacements(int global_size,
                                     int processes_count,
                                     std::vector<int>& counts,
                                     std::vector<int>& displacements) {
  const int base = global_size / processes_count;
  const int remainder = global_size % processes_count;

  int offset = 0;

  for (int i = 0; i < processes_count; ++i) {
    counts[i] = base + (i < remainder ? 1 : 0);
    displacements[i] = offset;
    offset += counts[i];
  }
}

std::vector<int> ChetverikovaEShellSortSimpleMergeALL::MergeLocalBuffers(
    std::vector<std::vector<int>>& local_buffers) {
  if (local_buffers.empty()) {
    return {};
  }

  std::vector<int> result = std::move(local_buffers[0]);

  for (size_t i = 1; i < local_buffers.size(); ++i) {
    result = MergeTwoSortedVectors(result, local_buffers[i]);
  }

  return result;
}


bool ChetverikovaEShellSortSimpleMergeALL::RunImpl() {
  int rank = 0;
  int processes_count = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);

  auto& output = GetOutput();

  int global_size = 0;

  if (rank == 0) {
    global_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(processes_count);
  std::vector<int> displacements(processes_count);

  if (rank == 0) {
    CalculateCountsAndDisplacements(global_size, processes_count,
                                    sendcounts, displacements);
  }

  int local_size = 0;

  MPI_Scatter(rank == 0 ? sendcounts.data() : nullptr,
              1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(local_size);

  const int* send_buffer =
      rank == 0 ? GetInput().data() : nullptr;

  MPI_Scatterv(send_buffer,
               rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displacements.data() : nullptr,
               MPI_INT, local_data.data(), local_size,
               MPI_INT, 0, MPI_COMM_WORLD);

  const std::size_t threads =
      std::max(1, omp_get_max_threads());

  const std::size_t parts =
      std::max<std::size_t>(
          1,
          std::min<std::size_t>(threads, local_data.size()));

  std::vector<std::vector<int>> local_buffers(parts);

  std::vector<size_t> indices;
  indices.reserve(parts + 1);
  indices.push_back(0);

  const size_t block = local_data.size() / parts;
  const size_t remainder = local_data.size() % parts;

  for (size_t i = 0; i < parts; ++i) {
    size_t next = indices.back() + block;

    if (i < remainder) {
      ++next;
    }

    indices.push_back(next);
  }

#pragma omp parallel for default(none) \
    shared(local_data, local_buffers, indices, parts) \
    schedule(static)
  for (size_t i = 0; i < parts; ++i) {
    const size_t left = indices[i];
    const size_t right = indices[i + 1];

    std::vector<int> temp(
        local_data.begin() + static_cast<std::ptrdiff_t>(left),
        local_data.begin() + static_cast<std::ptrdiff_t>(right));

    ShellSort(temp);

    local_buffers[i] = std::move(temp);
  }

  std::vector<int> local_sorted =
      MergeLocalBuffers(local_buffers);

  const int local_sorted_size =
      static_cast<int>(local_sorted.size());

  std::vector<int> recvcounts(processes_count);
  std::vector<int> recvdisplacements(processes_count);

  MPI_Gather(&local_sorted_size, 1, MPI_INT,
             rank == 0 ? recvcounts.data() : nullptr,
             1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_size = 0;

  if (rank == 0) {
    for (int i = 0; i < processes_count; ++i) {
      recvdisplacements[i] = total_size;
      total_size += recvcounts[i];
    }

    output.resize(total_size);
  }

  MPI_Gatherv(local_sorted.data(),
              local_sorted_size, MPI_INT,
              rank == 0 ? output.data() : nullptr,
              rank == 0 ? recvcounts.data() : nullptr,
              rank == 0 ? recvdisplacements.data() : nullptr,
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<std::vector<int>> gathered_parts(processes_count);

    for (int i = 0; i < processes_count; ++i) {
      const int begin = recvdisplacements[i];
      const int end = begin + recvcounts[i];

      gathered_parts[i] = std::vector<int>(
          output.begin() + begin,
          output.begin() + end);
    }

    output = std::move(gathered_parts[0]);

    for (int i = 1; i < processes_count; ++i) {
      output = MergeTwoSortedVectors(output, gathered_parts[i]);
    }
  }

  return true;
}

bool ChetverikovaEShellSortSimpleMergeALL::PostProcessingImpl() {
  int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int output_size = 0;

  if (rank == 0) {
    output_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(output_size);
  }

  MPI_Bcast(GetOutput().data(), output_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace chetverikova_e_shell_sort_simple_merge
