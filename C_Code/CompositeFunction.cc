#include "CompositeFunction.h"

#include "special.h"

CompositeFunction::CompositeFunction(double lambda1)
    : lambda1(lambda1)
{
}

/* returns argmin_x (A * lambda1||x||_1 + 1/2||x - x_0||) */
Eigen::VectorXd CompositeFunction::prox1(const Eigen::VectorXd& x0, double A) const
{
    return soft_threshold(x0, A * lambda1);
}
