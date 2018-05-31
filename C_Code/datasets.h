#ifndef DATASETS_H
#define DATASETS_H

#include <Eigen/Dense>

void load_mushrooms(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_a9a(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_w8a(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_covtype(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_cod_rna(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_ijcnn1(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_SUSY(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_mnist(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_mnist8m(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_gisette(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_quantum(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_protein(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_alpha(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_epsilon(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_zeta(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_beta(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_gamma(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_delta(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_fd(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_ocr(Eigen::MatrixXd& X, Eigen::VectorXi& y);
void load_dna18(Eigen::MatrixXd& X, Eigen::VectorXi& y);

#endif
