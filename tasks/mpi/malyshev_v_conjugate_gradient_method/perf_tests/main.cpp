#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

TEST(malyshev_v_conjugate_gradient_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  int numRowsA = 512;
  int numColsA = 512;

  std::vector<double> A(numRowsA * numColsA, 0.0);
  for (int i = 0; i < numRowsA * numColsA; i++) {
    A[i] = i % 100 + 1;
  }
  std::vector<double> b(numRowsA, 0.0);
  for (int i = 0; i < numRowsA; i++) {
    b[i] = i % 50 + 1;
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<double> x(numRowsA);
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  }
  auto testTask = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int numRowsA = 512;
  int numColsA = 512;

  std::vector<double> A(numRowsA * numColsA, 0.0);
  for (int i = 0; i < numRowsA * numColsA; i++) {
    A[i] = i % 100 + 1;
  }

  std::vector<double> b(numRowsA, 0.0);
  for (int i = 0; i < numRowsA; i++) {
    b[i] = i % 50 + 1;
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<double> x(numRowsA);
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  }

  auto testTask = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}