#include <iostream>
#include <cstdio>
#include <cstring>

// #include <algorithm>    // std::max

#include <tclap/CmdLine.h>
#include <Eigen/Dense>

#include "datasets.h"
#include "LogRegOracle.h"
#include "optim.h"
#include "Logger.h"

int main(int argc, char* argv[])
{
    /* ============================= Parse commmand-line arguments ==================================== */
    std::string method = "";
    std::string dataset = "";
    std::string sampling_scheme = "";
    std::string init_scheme = "";
    std::string output_filename = "";
    double lambda = -1;
    double epoch_size = 1.0;
    int epoch = 1;
    int minibatch_size = 1;
    double max_epochs = 1.0;
    double n_logs_per_epoch = -1;
    double alpha = -1;
    double beta = -1;
    double kappa = -1;
    double tol = 1e-10;
    double opt_allowed_time = -1;
    bool exact = false;

    try {
        /* prepare parser */
        TCLAP::CmdLine cmd("Run a numerical optimiser for training logistic regression.", ' ', "0.1");

        /* specify all options */
        TCLAP::ValueArg<std::string> arg_method(
            "", "method",
            "Optimisation method (SGD, SAG, NIM, newton, HFN, LBFGS)",
            true, method, "string"
        );
        TCLAP::ValueArg<std::string> arg_dataset(
            "", "dataset",
            "Dataset (a9a, mushrooms, w8a, covtype, cod-rna, ijcnn1, gisette, quantum, protein, alpha, beta, "
            "gamma, delta, epsilon, zeta, fd, ocr, dna18)",
            true, dataset, "string"
        );
        TCLAP::ValueArg<double> arg_lambda(
            "", "lambda",
            "L2-regularisation coefficient (default: 1/N)",
            false, lambda, "double"
        );
        TCLAP::ValueArg<double> arg_minibatch_size(
            "", "minibatch_size",
            "Minibatch size (default: 1). This parameter has no effect on non-incremental methods.",
            false, minibatch_size, "int"
        );
        TCLAP::ValueArg<double> arg_epoch_size(
            "", "epoch_size",
            "Epoch size as multiples of datasize (default: 1.0)",
            false, epoch_size, "double"
        );
        TCLAP::ValueArg<double> arg_max_epochs(
            "", "max_epochs",
            "Maximum number of epochs (default: 1.0)",
            false, max_epochs, "double"
        );
        TCLAP::ValueArg<double> arg_n_logs_per_epoch(
            "", "n_logs_per_epoch",
            "Number of requested logs per epoch (default: 1.0 for SGD and SAG; 10.0 for NIM)",
            false, n_logs_per_epoch, "double"
        );
        TCLAP::ValueArg<double> arg_alpha(
            "", "alpha",
            "Step length for incremental methods (or learning rate for SGD) (default: 1.0 for NIM or SGD; "
            "1/L for SAG where L is the (global) Lipschitz constant)",
            false, alpha, "double"
        );
        TCLAP::ValueArg<double> arg_beta(
            "", "beta",
            "Extrapolation Weight for Accelerated Methods",
            false, beta, "double"
        );
        TCLAP::ValueArg<double> arg_kappa(
            "", "kappa",
            "Step Size Scaling Parameter",
            false, kappa, "double"
        );
        TCLAP::ValueArg<double> arg_tol(
            "", "tol",
            "Gradient norm tolerance (default: 1e-10)",
            false, tol, "double"
        );
        TCLAP::ValueArg<double> arg_opt_allowed_time(
            "", "opt_allowed_time",
            "Maximal amount of time for which the optimiser is allowed to work; set -1 for no limit (default: -1)",
            false, opt_allowed_time, "double"
        );
        TCLAP::ValueArg<std::string> arg_sampling_scheme(
            "", "sampling_scheme",
            "Sampling scheme: cyclic, random or permute (only for incremental methods) "
            "(default: random for SAG and SGD; cyclic for NIM)",
            false, sampling_scheme, "string"
        );
        TCLAP::ValueArg<std::string> arg_init_scheme(
            "", "init_scheme",
            "Initialisation scheme (only for SAG or NIM): self-init (default) or full (initialise every component at w0)",
            false, init_scheme, "string"
        );
        TCLAP::ValueArg<std::string> arg_output_filename(
            "", "output_filename",
            "Filename for the output file (if not specified, will be generated automatically)",
            false, output_filename, "string"
        );
        TCLAP::ValueArg<bool> arg_exact(
            "", "exact",
            "Solve subprolem exactly (accuracy 1e-10) or not (only for NIM and Newton): true or false",
            false, exact, "bool"
        );

        /* add options to parser */
        cmd.add(arg_exact);
        cmd.add(arg_output_filename);
        cmd.add(arg_init_scheme);
        cmd.add(arg_sampling_scheme);
        cmd.add(arg_opt_allowed_time);
        cmd.add(arg_tol);
        cmd.add(arg_n_logs_per_epoch);
        cmd.add(arg_alpha);
        cmd.add(arg_beta);
        cmd.add(arg_kappa);
        cmd.add(arg_max_epochs);
        cmd.add(arg_minibatch_size);
        cmd.add(arg_epoch_size);
        cmd.add(arg_lambda);
        cmd.add(arg_dataset);
        cmd.add(arg_method);

        /* parse command-line string */
        cmd.parse(argc, argv);

        /* retrieve option values */
        method = arg_method.getValue();
        dataset = arg_dataset.getValue();
        lambda = arg_lambda.getValue();
        minibatch_size = arg_minibatch_size.getValue();
        epoch_size = arg_epoch_size.getValue();
        max_epochs = arg_max_epochs.getValue();
        n_logs_per_epoch = arg_n_logs_per_epoch.getValue();
        alpha = arg_alpha.getValue();
        beta = arg_beta.getValue();
        kappa = arg_kappa.getValue();
        tol = arg_tol.getValue();
        opt_allowed_time = arg_opt_allowed_time.getValue();
        sampling_scheme = arg_sampling_scheme.getValue();
        init_scheme = arg_init_scheme.getValue();
        output_filename = arg_output_filename.getValue();
        exact = arg_exact.getValue();
    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    /* ============================= Load dataset ==================================== */

    Eigen::MatrixXd Z;
    Eigen::VectorXi y;

    if (dataset == "a9a") {
        fprintf(stderr, "Load a9a\n");
        load_a9a(Z, y);
    } else if (dataset == "mushrooms") {
        fprintf(stderr, "Load mushrooms\n");
        load_mushrooms(Z, y);
    } else if (dataset == "w8a") {
        fprintf(stderr, "Load w8a\n");
        load_w8a(Z, y);
    } else if (dataset == "covtype") {
        fprintf(stderr, "Load covtype\n");
        load_covtype(Z, y);
    } else if (dataset == "cod-rna") {
        fprintf(stderr, "Load cod-rna\n");
        load_cod_rna(Z, y);
    } else if (dataset == "ijcnn1") {
        fprintf(stderr, "Load ijcnn1\n");
        load_ijcnn1(Z, y);
    } else if (dataset == "SUSY") {
        fprintf(stderr, "Load SUSY\n");
        load_SUSY(Z, y);
    } else if (dataset == "mnist8m") {
        fprintf(stderr, "Load mnist8m\n");
        load_mnist8m(Z, y);
    } else if (dataset == "mnist") {
        fprintf(stderr, "Load mnist\n");
        load_mnist(Z, y);
    } else if (dataset == "gisette") {
        fprintf(stderr, "Load gisette. It may take some time.\n");
        load_gisette(Z, y);
    } else if (dataset == "quantum") {
        fprintf(stderr, "Load quantum\n");
        load_quantum(Z, y);
    } else if (dataset == "protein") {
        fprintf(stderr, "Load protein\n");
        load_protein(Z, y);
    } else if (dataset == "alpha") {
        fprintf(stderr, "Load alpha, might take a lot of time\n");
        load_alpha(Z, y);
    } else if (dataset == "epsilon") {
        fprintf(stderr, "Load epsilon. It may take a while.\n");
        load_epsilon(Z, y);
    } else if (dataset == "zeta") {
        fprintf(stderr, "Load zeta. It may take a while.\n");
        load_zeta(Z, y);
    } else if (dataset == "beta") {
        fprintf(stderr, "Load beta. It may take a while.\n");
        load_beta(Z, y);
    } else if (dataset == "gamma") {
        fprintf(stderr, "Load gamma. It may take a while.\n");
        load_gamma(Z, y);
    } else if (dataset == "delta") {
        fprintf(stderr, "Load delta. It may take a while.\n");
        load_delta(Z, y);
    } else if (dataset == "fd") {
        fprintf(stderr, "Load fd. It may take a while.\n");
        load_fd(Z, y);
    } else if (dataset == "ocr") {
        fprintf(stderr, "Load ocr. It may take a while.\n");
        load_ocr(Z, y);
    } else if (dataset == "dna18") {
        fprintf(stderr, "Load dna18. It may take a while.\n");
        load_dna18(Z, y);
    } else {
        fprintf(stderr, "Unknown dataset %s\n", dataset.c_str());
        return 1;
    }

    /* ============================= Construct matrix Z ==================================== */

    Z.array().colwise() *= -y.cast<double>().array(); // multiply each sample X[i] by -y[i]

    /* ============================= Set up parameters ==================================== */

    /* starting point */
    Eigen::VectorXd w0 = Eigen::VectorXd::Zero(Z.cols()); // start from zero
    /* regularisation coefficient */
    if (lambda == -1) { // if not set up yet
        lambda = 1.0 / Z.rows();
    }
    /* number of logs per epoch */
    if (n_logs_per_epoch == -1) { // if not set up yet
        if (method == "NIM" || method == "CIAG" || method == "ACIAG" ) {
            n_logs_per_epoch = 10.0;
        } else if (method == "LBFGS") {
            n_logs_per_epoch = 0.25;
        } else if (method == "FGD") {
            n_logs_per_epoch = 0.005;
        } else {
            n_logs_per_epoch = 1;
        }
    }
    /* maximum number of iterations */
    size_t maxiter;
    if (method == "SGD" || method == "SAG" || method == "ASVRG" || method == "SAGA" || method == "NIM" || method == "CIAG" || method == "ACIAG" ) { // incremental methods
        maxiter = max_epochs * size_t(ceil(double(Z.rows()) / minibatch_size));
    } else { // non-incremental methods, one iteration >= one epoch
        maxiter = max_epochs;
    }
    /* step length */
    double L = 0.25 * Z.rowwise().squaredNorm().sum() / ((double) Z.rows()) + lambda; // global Lipschitz constant
    if (kappa == -1) {
      if (method == "ACIAG" || method == "CIAG") {
        kappa = 0.0001;
      }
    }
    if (alpha == -1) { // if not set up yet
        if (method == "SAG") { // use alpha=1/L by default
            alpha = 1.0 / L;
        } else if (method == "SAGA" || method == "ASVRG") {
            alpha = 0.5 / L;
        } else if (method == "CIAG" || method == "ACIAG" || method == "FGD" ) {
          // alpha = (2.0 / (L + lambda));
          alpha =  Z.rows() * kappa / L;
        } else {
            alpha = 1.0;
        }
    }
    if (beta == -1) {
      if (method == "ACIAG") {
        beta = 0.99;
      }
    }
    /* sampling scheme */
    if (sampling_scheme == "") { // if not set up yet
        if (method == "SAG" || method == "SAGA" || method == "SGD") {
            sampling_scheme = "random";
        } else {
            sampling_scheme = "cyclic";
        }
    }
    /* initialisation scheme */
    if (init_scheme == "") { // if not set up yet
        init_scheme = "self-init";
    }
    /* epoch size */
    if (method == "ASVRG") {
        epoch = round( epoch_size*Z.rows() );
        beta = 0.9;
        fprintf(stderr, "epoch size = %d\n", epoch);
    }

    /* =============================== Run optimiser ======================================= */

    LogRegOracle func(Z, lambda, minibatch_size); // prepare oracle
    Logger logger(func, n_logs_per_epoch, tol, opt_allowed_time); // prepare logger

    fprintf(stderr, "lambda=%g, minibatch_size=%d, L=%g, max_epochs=%g\n", lambda, minibatch_size, L, max_epochs);
    /* run chosen method */
    if (method == "SAG") {
        /* print summary */
        fprintf(stderr, "Use method SAG: alpha=%g, sampling_scheme=%s, init_scheme=%s\n",
                alpha, sampling_scheme.c_str(), init_scheme.c_str());

        /* rum method */
        SAG(func, logger, w0, maxiter, alpha, sampling_scheme, init_scheme);
    } else if (method == "SAGA") {
        /* print summary */
        fprintf(stderr, "Use method SAGA: alpha=%g, sampling_scheme=%s, init_scheme=%s\n",
                alpha, sampling_scheme.c_str(), init_scheme.c_str());

        /* rum method */
        SAGA(func, logger, w0, maxiter, alpha, sampling_scheme, init_scheme);
    } else if (method == "ASVRG") {
        /* print summary */
        fprintf(stderr, "Use method ASVRG: alpha=%g, sampling_scheme=%s, init_scheme=%s\n",
                alpha, sampling_scheme.c_str(), init_scheme.c_str());

        /* rum method */
        ASVRG(func, logger, w0, maxiter, epoch, alpha, beta, sampling_scheme, init_scheme);
    } else if (method == "SGD") {
        /* print summary */
        fprintf(stderr, "Use method SGD: alpha=%g, sampling_scheme=%s\n", alpha, sampling_scheme.c_str());

        /* run method */
        SGD(func, logger, w0, maxiter, alpha, sampling_scheme);
    } else if (method == "NIM") {
        /* print summary */
        fprintf(stderr, "Use method NIM: alpha=%g, sampling_scheme=%s, init_scheme=%s, exact=%d\n",
                alpha, sampling_scheme.c_str(), init_scheme.c_str(), exact);

        /* run method */
        NIM(func, logger, w0, maxiter, alpha, sampling_scheme, init_scheme, exact);
    } else if (method == "CIAG") {
        /* print summary */
        fprintf(stderr, "Use method CIAG: alpha=%g, sampling_scheme=%s, init_scheme=%s\n",
                alpha, sampling_scheme.c_str(), init_scheme.c_str());

        /* run method */
        CIAG(func, logger, w0, maxiter, alpha, sampling_scheme, init_scheme);
    } else if (method == "ACIAG") {
        /* print summary */
        fprintf(stderr, "Use method ACIAG: alpha=%g, beta=%g, sampling_scheme=%s, init_scheme=%s\n",
                alpha, beta, sampling_scheme.c_str(), init_scheme.c_str());
        /* run method */
        ACIAG(func, logger, w0, maxiter, alpha, beta, sampling_scheme, init_scheme);
    } else if (method == "FGD") {
        /* print summary */
        fprintf(stderr, "Use method FGD: alpha=%g\n", alpha);
        /* run method */
        FGD(func, logger, w0, maxiter, alpha);
    } else if (method == "newton") {
        /* print summary */
        fprintf(stderr, "Use Newton's method: exact=%d\n", exact);

        /* run method */
        newton(func, logger, w0, maxiter, exact);
    } else if (method == "HFN") {
        /* print summary */
        fprintf(stderr, "Use method HFN\n");

        /* run method */
        HFN(func, logger, w0, maxiter);
    } else if (method == "LBFGS") {
        /* print summary */
        fprintf(stderr, "Use method L-BFGS\n");

        /* run method */
        LBFGS(func, logger, w0, maxiter);
    } else {
        fprintf(stderr, "Unknown method %s\n", method.c_str());
        return 1;
    }

    /* =============================== Print the trace ======================================= */

    /* construct the name of the output file */
    char out_filename[100];
    if (output_filename != "") {
        sprintf(out_filename, "output/%s", output_filename.c_str());
    } else {
        std::string prefix = lambda ? "l2" : "l1";
        if (method == "SAG" || method == "SAGA" || method == "ASVRG" || method == "SGD" || method == "NIM" || method == "CIAG" || method == "ACIAG" ) { // incremental methods
          if ( method == "NIM" ) {
            sprintf(out_filename, "output/%s.%s.%s.minibatch_size=%d.exact=%d.dat", prefix.c_str(), dataset.c_str(), method.c_str(), minibatch_size, exact);
          } else {
            sprintf(out_filename, "output/%s.%s.%s.minibatch_size=%d.%s.dat", prefix.c_str(), dataset.c_str(), method.c_str(), minibatch_size,sampling_scheme.c_str());
          }
        } else { // non-incremental methods
            if (method == "newton" ) { // can be exact or inexact
               sprintf(out_filename, "output/%s.%s.%s.exact=%d.dat", prefix.c_str(), dataset.c_str(), method.c_str(), exact);
            } else {
               sprintf(out_filename, "output/%s.%s.%s.dat", prefix.c_str(), dataset.c_str(), method.c_str());
            }
        }
    }

    /* creare output file */
    FILE* out_file;
    if (!(out_file = fopen(out_filename, "w"))) {
        fprintf(stderr, "Could not open output file '%s': %s\n", out_filename, strerror(errno));
        return 1;
    }

    /* write trace into it */
    fprintf(out_file, "%9s %9s %25s %25s\n", "epoch", "elapsed", "val", "norm_grad");
    for (size_t i = 0; i < logger.trace_epoch.size(); ++i) {
        fprintf(out_file, "%9.2f %9.3f %25.16e %25.16e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
    }

    /* close output file */
    fclose(out_file);

    return 0;
}
