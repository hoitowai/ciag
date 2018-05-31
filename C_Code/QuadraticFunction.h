#ifndef QUADRATIC_FUNCTION_H_
#define QUADRATIC_FUNCTION_H_

#include <Eigen/Dense>

#include "CompositeFunction.h"

class QuadraticFunction : public CompositeFunction
{
public:
    QuadraticFunction(const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double lambda1);

    Eigen::VectorXd full_grad(const Eigen::VectorXd& x) const;

    const Eigen::MatrixXd& Q;
    const Eigen::VectorXd& b;
};

#endif
