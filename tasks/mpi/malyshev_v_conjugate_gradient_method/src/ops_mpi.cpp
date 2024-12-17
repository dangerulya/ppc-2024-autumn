#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  numRowsA_ = taskData->inputs_count[0];
  numColsA_ = taskData->inputs_count[1];

  auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(a_data, a_data + (numRowsA_ * numColsA_));
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.assign(b_data, b_data + numRowsA_);

  x_.resize(numRowsA_, 0.0);
  r_.resize(numRowsA_, 0.0);
  p_.resize(numRowsA_, 0.0);

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 2) return false;

  int temp_numRowsA = taskData->inputs_count[0];
  int temp_numColsA = taskData->inputs_count[1];

  return (temp_numRowsA > 0 && temp_numColsA > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  // Initialize vectors
  std::vector<double> Ap(numRowsA_, 0.0);
  for (int i = 0; i < numRowsA_; i++) {
    r_[i] = b_[i];
    p_[i] = r_[i];
  }

  double r_norm_sq = 0.0;
  for (int i = 0; i < numRowsA_; i++) {
    r_norm_sq += r_[i] * r_[i];
  }

  for (int iter = 0; iter < numRowsA_; iter++) {
    // Compute Ap = A * p
    std::fill(Ap.begin(), Ap.end(), 0.0);
    for (int i = 0; i < numRowsA_; i++) {
      for (int j = 0; j < numColsA_; j++) {
        Ap[i] += A_[i * numColsA_ + j] * p_[j];
      }
    }

    // Compute alpha = r_norm_sq / (p' * Ap)
    double p_dot_Ap = 0.0;
    for (int i = 0; i < numRowsA_; i++) {
      p_dot_Ap += p_[i] * Ap[i];
    }
    double alpha = r_norm_sq / p_dot_Ap;

    // Update x and r
    for (int i = 0; i < numRowsA_; i++) {
      x_[i] += alpha * p_[i];
      r_[i] -= alpha * Ap[i];
    }

    // Compute new r_norm_sq
    double r_norm_sq_new = 0.0;
    for (int i = 0; i < numRowsA_; i++) {
      r_norm_sq_new += r_[i] * r_[i];
    }

    if (r_norm_sq_new < 1e-10) break;

    // Update p
    double beta = r_norm_sq_new / r_norm_sq;
    for (int i = 0; i < numRowsA_; i++) {
      p_[i] = r_[i] + beta * p_[i];
    }

    r_norm_sq = r_norm_sq_new;
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), data_ptr);

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank == 0) {
    numRowsA_ = taskData->inputs_count[0];
    numColsA_ = taskData->inputs_count[1];
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
    A_.assign(a_data, a_data + (numRowsA_ * numColsA_));
    auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
    b_.assign(b_data, b_data + numRowsA_);
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank != 0) return true;
  if (taskData->inputs_count.size() < 2) return false;

  int temp_numRowsA = taskData->inputs_count[0];
  int temp_numColsA = taskData->inputs_count[1];

  return (temp_numRowsA > 0 && temp_numColsA > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int size;
  int rank;
  MPI_Comm_size(world, &size);
  MPI_Comm_rank(world, &rank);

  MPI_Bcast(&numRowsA_, 1, MPI_INT, 0, world);
  MPI_Bcast(&numColsA_, 1, MPI_INT, 0, world);

  int q = static_cast<int>(std::floor(std::sqrt(size)));
  int active_procs = q * q;

  int padded_m = q * ((numRowsA_ + q - 1) / q);
  int padded_n = q * ((numColsA_ + q - 1) / q);

  int block_m = padded_m / q;
  int block_n = padded_n / q;

  if (rank == 0) {
    std::vector<double> A_padded(padded_m * padded_n, 0.0);
    for (int i = 0; i < numRowsA_; i++) {
      std::copy(A_.begin() + i * numColsA_, A_.begin() + (i + 1) * numColsA_,
                A_padded.begin() + i * padded_n);
    }
    A_ = std::move(A_padded);
  }

  bool is_active = (rank < active_procs);
  int color = is_active ? 1 : MPI_UNDEFINED;
  MPI_Comm active_comm;
  MPI_Comm_split(world, color, rank, &active_comm);

  if (!is_active) {
    return true;
  }

  MPI_Comm cart_comm;
  int dims_grid[2] = {q, q};
  int periods_grid[2] = {1, 1};
  int reorder_grid = 0;
  MPI_Cart_create(active_comm, 2, dims_grid, periods_grid, reorder_grid, &cart_comm);

  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);
  int coords[2];
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
  int row = coords[0];
  int col = coords[1];

  int left_rank;
  int right_rank;
  int up_rank;
  int down_rank;
  MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);
  MPI_Cart_shift(cart_comm, 0, 1, &up_rank, &down_rank);

  std::vector<double> A_local(block_m * block_n, 0.0);
  std::vector<double> b_local(block_m, 0.0);
  std::vector<double> x_local(block_m, 0.0);
  std::vector<double> r_local(block_m, 0.0);
  std::vector<double> p_local(block_n, 0.0);

  if (cart_rank == 0) {
    for (int p = 0; p < active_procs; ++p) {
      int p_coords[2];
      MPI_Cart_coords(cart_comm, p, 2, p_coords);
      int p_row = p_coords[0];
      int p_col = p_coords[1];

      std::vector<double> A_block(block_m * block_n, 0.0);
      for (int i = 0; i < block_m; i++) {
        int src_index = (p_row * block_m + i) * padded_n + (p_col * block_n);
        std::copy(A_.begin() + src_index, A_.begin() + src_index + block_n, A_block.begin() + i * block_n);
      }
      std::vector<double> b_block(block_m, 0.0);
      std::copy(b_.begin() + p_row * block_m, b_.begin() + (p_row + 1) * block_m, b_block.begin());

      if (p == 0) {
        A_local = std::move(A_block);
        b_local = std::move(b_block);
      } else {
        MPI_Send(A_block.data(), block_m * block_n, MPI_DOUBLE, p, 0, cart_comm);
        MPI_Send(b_block.data(), block_m, MPI_DOUBLE, p, 1, cart_comm);
      }
    }
  } else {
    MPI_Recv(A_local.data(), block_m * block_n, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Recv(b_local.data(), block_m, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
  }

  // Initialize vectors
  for (int i = 0; i < block_m; i++) {
    r_local[i] = b_local[i];
    p_local[i] = r_local[i];
  }

  double r_norm_sq = 0.0;
  for (int i = 0; i < block_m; i++) {
    r_norm_sq += r_local[i] * r_local[i];
  }

  for (int iter = 0; iter < numRowsA_; iter++) {
    // Compute Ap = A * p
    std::vector<double> Ap_local(block_m, 0.0);
    for (int i = 0; i < block_m; i++) {
      for (int j = 0; j < block_n; j++) {
        Ap_local[i] += A_local[i * block_n + j] * p_local[j];
      }
    }

    // Compute alpha = r_norm_sq / (p' * Ap)
    double p_dot_Ap = 0.0;
    for (int i = 0; i < block_m; i++) {
      p_dot_Ap += p_local[i] * Ap_local[i];
    }
    double alpha = r_norm_sq / p_dot_Ap;

    // Update x and r
    for (int i = 0; i < block_m; i++) {
      x_local[i] += alpha * p_local[i];
      r_local[i] -= alpha * Ap_local[i];
    }

    // Compute new r_norm_sq
    double r_norm_sq_new = 0.0;
    for (int i = 0; i < block_m; i++) {
      r_norm_sq_new += r_local[i] * r_local[i];
    }

    if (r_norm_sq_new < 1e-10) break;

    // Update p
    double beta = r_norm_sq_new / r_norm_sq;
    for (int i = 0; i < block_m; i++) {
      p_local[i] = r_local[i] + beta * p_local[i];
    }

    r_norm_sq = r_norm_sq_new;
  }

  if (cart_rank != 0) {
    MPI_Send(x_local.data(), block_m, MPI_DOUBLE, 0, 2, cart_comm);
  } else {
    x_.resize(padded_m, 0.0);
    for (int i = 0; i < block_m; i++) {
      x_[i] = x_local[i];
    }
    for (int p = 1; p < active_procs; p++) {
      std::vector<double> recv_x(block_m);
      MPI_Recv(recv_x.data(), block_m, MPI_DOUBLE, p, 2, cart_comm, MPI_STATUS_IGNORE);

      int p_coords[2];
      MPI_Cart_coords(cart_comm, p, 2, p_coords);
      int p_row = p_coords[0];

      int start_row = p_row * block_m;
      for (int i = 0; i < block_m; i++) {
        x_[start_row + i] = recv_x[i];
      }
    }

    std::vector<double> x_final(numRowsA_, 0.0);
    for (int i = 0; i < numRowsA_; i++) {
      x_final[i] = x_[i];
    }
    x_ = std::move(x_final);
  }

  if (cart_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm);
  }
  MPI_Comm_free(&active_comm);

  return true;
}

bool malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank == 0) {
    auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(x_.begin(), x_.end(), data_ptr);
  }

  return true;
}