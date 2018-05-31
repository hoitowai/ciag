#ifndef SPECIAL_H
#define SPECIAL_H

#include <Eigen/Dense>

/* Calculate log(exp(a) + exp(b)) without floating-point overflows */
double logaddexp(double a, double b);
double logaddexp0(double x); // the same but for log(1 + exp(x))

/* Sigmoid function */
double sigm(double x);

/* Soft-thresholding operator */
Eigen::VectorXd soft_threshold(const Eigen::VectorXd& x0, double tau); // d-dimensional

#endif
