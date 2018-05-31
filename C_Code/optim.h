#ifndef OPTIM_H
#define OPTIM_H

#include <string>
#include <Eigen/Dense>

#include "CompositeFunction.h"
#include "LogRegOracle.h"
#include "Logger.h"

/* Method SGD */
Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme);

/* Method SAG for **linear models** */
Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme, const std::string& init_scheme);

/* Method NIM for **linear models** */
Eigen::VectorXd NIM(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme, const std::string& init_scheme, bool exact=false);

/* Method CIAG for **linear models** */
Eigen::VectorXd CIAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    const std::string& sampling_scheme, const std::string& init_scheme);

/* Method ACIAG for **linear models** */
Eigen::VectorXd ACIAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha,
                    double beta, const std::string& sampling_scheme, const std::string& init_scheme);

/* Newton's method (with unit step length) for a general strongly convex function */
Eigen::VectorXd FGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha);

/* Newton's method (with unit step length) for a general strongly convex function */
Eigen::VectorXd newton(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, bool exact=false);

/* Fast gradient method */
Eigen::VectorXd fgm(const CompositeFunction& func, const Eigen::VectorXd& x0, size_t maxiter, double tol=1e-5, double L0=1);

/* Conjugate gradient method */
Eigen::VectorXd cg(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
                   const Eigen::VectorXd& b, const Eigen::VectorXd& x0, double tol);

/* Hessian-free Newton method */
Eigen::VectorXd HFN(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1=1e-4);

/* Method L-BFGS for a general strongly convex function */
Eigen::VectorXd LBFGS(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, size_t m=10, double c1=1e-4);

#endif
