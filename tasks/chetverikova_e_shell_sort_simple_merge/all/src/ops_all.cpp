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

  std::merge(a, b, std::back_inserter(result));

  return result;
}

bool ChetverikovaEShellSortSimpleMergeALL::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &output = GetOutput();

  std::vector<int> local_data;

  int global_size = 0;

  if (rank == 0) {
    global_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    int base = global_size / size;
    int rem = global_size % size;

    int offset = 0;

    for (int i = 0; i < size; ++i) {
      sendcounts[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  int local_size = 0;

  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  local_data.resize(local_size);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  const std::size_t threads = std::max(1, omp_get_max_threads());

  const std::size_t parts = std::min<std::size_t>(threads, local_data.size());

  std::vector<std::vector<int>> local_buffers(parts);

  std::vector<size_t> indices;
  indices.push_back(0);

  const size_t block = local_data.size() / parts;
  const size_t rem = local_data.size() % parts;

  for (size_t i = 0; i < parts; ++i) {
    indices.push_back(indices.back() + block);

    if (i < rem) {
      indices[i + 1]++;
    }
  }

#pragma omp parallel for default(none) shared(local_data, local_buffers, indices, parts) schedule(static)
  for (size_t i = 0; i < parts; ++i) {
    size_t left = indices[i];
    size_t right = indices[i + 1];

    std::vector<int> temp(local_data.begin() + static_cast<std::ptrdiff_t>(left),
                          local_data.begin() + static_cast<std::ptrdiff_t>(right));

    ShellSort(temp);

    local_buffers[i] = std::move(temp);
  }

  std::vector<int> local_sorted;

  if (!local_buffers.empty()) {
    local_sorted = std::move(local_buffers[0]);

    for (size_t i = 1; i < local_buffers.size(); ++i) {
      local_sorted = MergeTwoSortedVectors(local_sorted, local_buffers[i]);
    }
  }

  int local_sorted_size = static_cast<int>(local_sorted.size());

  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);

  MPI_Gather(&local_sorted_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_size = 0;

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      recvdispls[i] = total_size;
      total_size += recvcounts[i];
    }
    output.resize(total_size);
  }

  MPI_Gatherv(local_sorted.data(), local_sorted_size, MPI_INT, rank == 0 ? output.data() : nullptr, recvcounts.data(),
              recvdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<std::vector<int>> gathered_parts(size);

    for (int i = 0; i < size; ++i) {
      int begin = recvdispls[i];
      int end = begin + recvcounts[i];

      gathered_parts[i] = std::vector<int>(output.begin() + begin, output.begin() + end);
    }

    output = std::move(gathered_parts[0]);

    for (int i = 1; i < size; ++i) {
      output = MergeTwoSortedVectors(output, gathered_parts[i]);
    }
  }

  return true;
}

bool ChetverikovaEShellSortSimpleMergeALL::PostProcessingImpl() {
  /*int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int output_size = 0;

  if (rank == 0) {
    output_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(output_size);
  }

  MPI_Bcast(GetOutput().data(), output_size, MPI_INT, 0, MPI_COMM_WORLD);*/

  return true;
}

}  // namespace chetverikova_e_shell_sort_simple_merge
