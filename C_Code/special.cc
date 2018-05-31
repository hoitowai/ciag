#include <algorithm>
#include <cmath>

#include "special.h"

double logaddexp(double a, double b)
{
    double t = std::max(a, b);
    return t + log(exp(a - t) + exp(b - t));
}

double logaddexp0(double x)
{
    return logaddexp(0.0, x);
}

double sigm(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/* returns argmin_x (tau||x||_1 + 1/2||x - x_0||) */
Eigen::VectorXd soft_threshold(const Eigen::VectorXd& x0, double tau)
{
    //Eigen::VectorXd x = Eigen::VectorXd::Zero(x0.size());
    //x = (x0.array() > tau).select(x0.array() - tau, x);
    //x = (x0.array() < -tau).select(x0.array() + tau, x);
    Eigen::VectorXd x = x0;
    for (int i = 0; i < int(x0.size()); ++i) {
        if (x[i] < -tau) {
            x[i] += tau;
        } else if (x[i] > tau) {
            x[i] -= tau;
        } else {
            x[i] = 0;
        }
    }
    return x;
}
