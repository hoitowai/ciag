#include "QuadraticFunction.h"

QuadraticFunction::QuadraticFunction(const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double lambda1)
    : CompositeFunction(lambda1), Q(Q), b(b)
{
}

Eigen::VectorXd QuadraticFunction::full_grad(const Eigen::VectorXd& x) const
{
    return Q.selfadjointView<Eigen::Upper>() * x + b;
}
