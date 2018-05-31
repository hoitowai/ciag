#include <iostream>

#include "LogRegOracle.h"
#include "special.h"

LogRegOracle::LogRegOracle(const Eigen::MatrixXd& Z, double lambda, int minibatch_size)
    : CompositeFunction(0), Z(Z), lambda(lambda), minibatch_size(minibatch_size)
{
}

int LogRegOracle::get_n_minibatches() const
{
    return ceil(double(Z.rows()) / minibatch_size);
}

int LogRegOracle::get_jth_minibatch_size(int j) const
{
    int rem = Z.rows() % minibatch_size;
    if (rem == 0 || j < get_n_minibatches() - 1) {
        return minibatch_size;
    }
    return rem;
}

const Eigen::Block<const Eigen::MatrixXd> LogRegOracle::get_jth_submatrix(int j) const
{
    int size_j = get_jth_minibatch_size(j);
    return Z.block(j * minibatch_size, 0, size_j, Z.cols());
}

int LogRegOracle::n_samples() const { return Z.rows(); }

double LogRegOracle::single_val(const Eigen::VectorXd& w, int i) const
{
    return logaddexp(0, Z.row(i).dot(w)) + (lambda / 2) * w.squaredNorm();
}

Eigen::VectorXd LogRegOracle::single_grad(const Eigen::VectorXd& w, int i) const
{
    return sigm(Z.row(i).dot(w)) * Z.row(i).transpose() + lambda * w;
}

double LogRegOracle::full_val(const Eigen::VectorXd& w) const // This is the same as written in the paper
{
    return (1.0 / Z.rows()) * (Z * w).unaryExpr(std::ptr_fun(logaddexp0)).sum() + (lambda / 2) * w.squaredNorm();
}

Eigen::VectorXd LogRegOracle::full_grad(const Eigen::VectorXd& w) const
{
    return (1.0 / Z.rows()) * Z.transpose() * (Z * w).unaryExpr(std::ptr_fun(sigm)) + lambda * w;
}

Eigen::MatrixXd LogRegOracle::full_hess(const Eigen::VectorXd& w) const
{
    /* calcuate the diagonal part */
    Eigen::VectorXd sigma = (Z * w).unaryExpr(std::ptr_fun(sigm));
    Eigen::VectorXd s = sigma.array() * (1.0 - sigma.array());

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(w.size(), w.size());
    /* since we don't want to copy the entire Z matrix, the only way to compute Z.T*S*Z
       without allocating new memory is to loop over all the samples */
    for (int i = 0; i < n_samples(); ++i) {
        H.selfadjointView<Eigen::Upper>().rankUpdate(Z.row(i).transpose(), s(i));
    }
    //Eigen::MatrixXd ZS = Z.array().colwise() * s.unaryExpr(std::ptr_fun(sqrt)).array();
    //H.selfadjointView<Eigen::Upper>().rankUpdate(ZS.transpose());
    H /= Z.rows(); // normalise by the number of samples

    /* don't forget the regulariser */
    H += lambda * Eigen::MatrixXd::Identity(w.size(), w.size());

    return H;
}

double LogRegOracle::full_val_grad(const Eigen::VectorXd& w, Eigen::VectorXd& g) const
{
    Eigen::VectorXd zw = Z * w;
    g = (1.0 / Z.rows()) * Z.transpose() * zw.unaryExpr(std::ptr_fun(sigm)) + lambda * w;
    return (1.0 / Z.rows()) * zw.unaryExpr(std::ptr_fun(logaddexp0)).sum() + (lambda / 2) * w.squaredNorm();
}

LogRegHessVec LogRegOracle::hessvec() const
{
    return LogRegHessVec(Z, lambda);
}

Eigen::VectorXd LogRegOracle::phi_prime(const Eigen::VectorXd& mu) const
{
    return mu.unaryExpr(std::ptr_fun(sigm));
}

Eigen::VectorXd LogRegOracle::phi_double_prime(const Eigen::VectorXd& mu) const
{
    Eigen::VectorXd s = mu.unaryExpr(std::ptr_fun(sigm));
    return s.array() * (1 - s.array());
}

/* ****************************************************************************************************************** */
/* ************************************************ LogRegHessVec *************************************************** */
/* ****************************************************************************************************************** */

LogRegHessVec::LogRegHessVec(const Eigen::MatrixXd& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

void LogRegHessVec::prepare(const Eigen::VectorXd& w)
{
    Eigen::VectorXd sigma = (Z * w).unaryExpr(std::ptr_fun(sigm));
    s = sigma.array() * (1.0 - sigma.array());
}

Eigen::VectorXd LogRegHessVec::calculate(const Eigen::VectorXd& d) const
{
    return (1.0 / Z.rows()) * Z.transpose() * (s.array() * (Z * d).array()).matrix() + lambda * d;
}
