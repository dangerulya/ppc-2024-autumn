#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

TEST(malyshev_v_conjugate_gradient_method_mpi, small_matrix_1x1) {
  boost::mpi::communicator world;
  int numRowsA = 1;
  int numColsA = 1;
  std::vector<double> A = {4};
  std::vector<double> b = {1};
  std::vector<double> expected_x = {0.25};
  std::vector<double> x_mpi(numRowsA, 0.0);
  std::vector<double> x_seq(numRowsA, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_mpi.data()));
  }

  malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));

    malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)x_mpi.size(); i++) {
      EXPECT_NEAR(x_mpi[i], expected_x[i], 1e-6);
      EXPECT_NEAR(x_mpi[i], x_seq[i], 1e-6);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, small_matrix_2x2) {
  boost::mpi::communicator world;
  int numRowsA = 2;
  int numColsA = 2;
  std::vector<double> A = {4, 1, 1, 3};
  std::vector<double> b = {1, 2};
  std::vector<double> expected_x = {0.0909091, 0.636364};
  std::vector<double> x_mpi(numRowsA, 0.0);
  std::vector<double> x_seq(numRowsA, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_mpi.data()));
  }

  malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));

    malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)x_mpi.size(); i++) {
      EXPECT_NEAR(x_mpi[i], expected_x[i], 1e-6);
      EXPECT_NEAR(x_mpi[i], x_seq[i], 1e-6);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, small_matrix_3x3) {
  boost::mpi::communicator world;
  int numRowsA = 3;
  int numColsA = 3;
  std::vector<double> A = {4, 1, 2, 1, 3, 1, 2, 1, 5};
  std::vector<double> b = {1, 2, 3};
  std::vector<double> expected_x = {0.0909091, 0.636364, 0.363636};
  std::vector<double> x_mpi(numRowsA, 0.0);
  std::vector<double> x_seq(numRowsA, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(numRowsA);
    taskData->inputs_count.emplace_back(numColsA);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_mpi.data()));
  }

  malyshev_v_conjugate_gradient_method_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_TRUE(testTask.validation());
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(numRowsA);
    taskDataSeq->inputs_count.emplace_back(numColsA);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_seq.data()));

    malyshev_v_conjugate_gradient_method_mpi::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)x_mpi.size(); i++) {
      EXPECT_NEAR(x_mpi[i], expected_x[i], 1e-6);
      EXPECT_NEAR(x_mpi[i], x_seq[i], 1e-6);
    }
  }
}