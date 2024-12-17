#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_conjugate_gradient_method_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  std::vector<double> r_;
  std::vector<double> p_;
  int numRowsA_;
  int numColsA_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  int numRowsA_;
  int numColsA_;
};

}  // namespace malyshev_v_conjugate_gradient_method_mpi