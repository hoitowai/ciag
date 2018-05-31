#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "LogRegOracle.h"

TEST(NSamplesTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_EQ(func.n_samples(), 3);
}

TEST(SingleValTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.single_val(w, 1), 0.721337293512, 1e-5);
}

TEST(SingleValTest, Overflow) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 1e5, 1e5;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.single_val(w, 2), 1000040000.0, 1e-5);
}

TEST(SingleGradTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    Eigen::VectorXd gi = func.single_grad(w, 1);
    EXPECT_NEAR(gi(0), 0.00490251, 1e-5);
    EXPECT_NEAR(gi(1), -0.09097488, 1e-5);
}

TEST(FullValTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.full_val(w), 0.70888955146, 1e-5);
}

TEST(FullValTest, Overflow) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 1e5, 1e5;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.full_val(w), 1000016333.333333, 1e-5);
}

TEST(FullGradTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    Eigen::VectorXd g = func.full_grad(w);
    EXPECT_NEAR(g(0), 0.07483417, 1e-5);
    EXPECT_NEAR(g(1), -0.04208662, 1e-5);
}

TEST(FullHessTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    Eigen::MatrixXd H = func.full_hess(w);
    /* because the Hessian is symmetric, require only the upper triangular part be correct */
    EXPECT_NEAR(H(0,0), 0.10834144, 1e-5);
    EXPECT_NEAR(H(0,1), 0.00249991, 1e-5);
    EXPECT_NEAR(H(1,1), 0.10167466, 1e-5);
}

TEST(LogRegHessVecTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w, d;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    d.resize(2);
    d << -0.2, 0.3;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    LogRegHessVec hv = func.hessvec();
    hv.prepare(w);

    Eigen::VectorXd hd = hv.calculate(d);
    EXPECT_NEAR(hd(0), -0.02091831, 1e-5);
    EXPECT_NEAR(hd(1), 0.03000242, 1e-5);
}
