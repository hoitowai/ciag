#ifndef COMPOSITE_FUNCTION_H_
#define COMPOSITE_FUNCTION_H_

#include <Eigen/Dense>

class CompositeFunction
{
public:
    CompositeFunction(double lambda1);

    Eigen::VectorXd prox1(const Eigen::VectorXd& x0, double A) const;
    virtual Eigen::VectorXd full_grad(const Eigen::VectorXd& w) const = 0;

    double lambda1; // l_1 regularization coefficient
};

#endif
