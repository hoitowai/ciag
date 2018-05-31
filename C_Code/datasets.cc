#include <fstream>
#include <iostream>
#include <set>
#include <cmath>
#include <cstring>

#include "datasets.h"

/* ================================================================================================================== */
/* ======================================== read_svmlight_file ====================================================== */
/* ================================================================================================================== */

void read_svmlight_file(const std::string& path, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* open file */
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "ERROR: Could not load file '%s'\n", path.c_str());
        throw 1;
    }

    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* loop over samples */
    std::string line;
    int how_often = round(0.05 * N); // how often print the progress
    for (int sample_idx = 0; sample_idx < N; ++sample_idx) {
        /* read current line */
        std::getline(file, line);

        /* display current progress */
        if (sample_idx % how_often == 0) {
            fprintf(stderr, "Processed %d/%d samples (%.2f%%)\n", sample_idx, N, round(double(sample_idx) / N * 100));
        }

        assert(sample_idx < N); // make sure the passed number of samples is correct

        int label, feature_idx;
        double feature_value;
        int offset;
        const char* cline = line.c_str();

        /* read label */
        sscanf(cline, "%d%n", &label, &offset);
        cline += offset;
        y(sample_idx) = label;

        /* read features */
        while (sscanf(cline, " %d:%lf%n", &feature_idx, &feature_value, &offset) == 2) {
            cline += offset;

            --feature_idx; // libsvm counts from 1
            assert(feature_idx >= 0 && feature_idx < D); // make sure the feature index is correct

            /* write this feature into the design matrix */
            X(sample_idx, feature_idx) = feature_value;
        }
    }

    /* we are done with the file, close it */
    file.close();
}

/* ================================================================================================================== */
/* ========================================= read_pascal_file ======================================================= */
/* ================================================================================================================== */

void read_pascal_file(const std::string& dat_filename, const std::string& lab_filename, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y,
    std::set<int> ignore_cols = std::set<int>())
{
    /* determine the actual number of features */
    int D_actual = D;
    if (!ignore_cols.empty()) { /* ignore the features in `ignore_cols` */
        D_actual -= ignore_cols.size();
    }

    /* allocate memory */
    X.resize(N, D_actual);
    y.resize(N);

    /* read design matrix X */
    int dummy;
    FILE* file;
    if (!(file = fopen(dat_filename.c_str(), "r"))) {
        fprintf(stderr, "Could not load file '%s': %s\n", dat_filename.c_str(), strerror(errno));
        throw 1;
    }
    int how_often = round(0.05 * N); // how often print the progress
    for (int i = 0; i < N; ++i) {
        /* display current progress */
        if (i % how_often == 0) {
            fprintf(stderr, "Processed %d/%d samples (%.2f%%)\n", i, N, round(double(i) / N * 100));
        }

        /* read features */
        double value;
        int j = 0;
        for (int j1 = 0; j1 < D; ++j1) {
            /* read feature value */
            dummy = fscanf(file, "%lf", &value);

            /* ignore this value if requested */
            if (ignore_cols.count(j1)) continue;

            /* otherwise, save it */
            X(i, j) = value;
            ++j;
        }
    }
    fclose(file);
    dummy += 0;

    /* read labels y */
    if (!(file = fopen(lab_filename.c_str(), "r"))) {
        fprintf(stderr, "Could not load file '%s': %s\n", lab_filename.c_str(), strerror(errno));
        throw 1;
    }
    for (int i = 0; i < N; ++i) {
        dummy = fscanf(file, "%d", &y(i));
    }
    fclose(file);
}

/* ================================================================================================================== */
/* ========================================== scale_features ======================================================== */
/* ================================================================================================================== */

void scale_features(Eigen::MatrixXd& X, int min, int max)
{
    /* compute min and max for each feature */
    Eigen::VectorXd features_min = X.colwise().minCoeff();
    Eigen::VectorXd features_max = X.colwise().maxCoeff();

    /* make sure no feature is constant */
    assert(((features_max - features_min).array() > 0.0).all());

    /* scale features to [0, 1] */
    X.array().rowwise() -= features_min.array().transpose();
    X.array().rowwise() /= (features_max - features_min).array().transpose();

    /* scale features to [min, max] */
    X = X.array() * (max - min) + min;
}

/* ****************************************************************************************************************** */
/* ********************************************** mushrooms ********************************************************* */
/* ****************************************************************************************************************** */

void load_mushrooms(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/mushrooms/mushrooms", 8124, 112, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

/* ****************************************************************************************************************** */
/* ************************************************* a9a ************************************************************ */
/* ****************************************************************************************************************** */

void load_a9a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/a9a/a9a", 32561, 123, X, y);
}

/* ****************************************************************************************************************** */
/* ************************************************* w8a ************************************************************ */
/* ****************************************************************************************************************** */

void load_w8a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/w8a/w8a", 49749 , 300, X, y);
}

/* ****************************************************************************************************************** */
/* *********************************************** covtype ********************************************************** */
/* ****************************************************************************************************************** */

void load_covtype(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/covtype/covtype.libsvm.binary.scale", 581012 , 54, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

/* ****************************************************************************************************************** */
/* *********************************************** cod-rna ********************************************************** */
/* ****************************************************************************************************************** */

void load_cod_rna(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/cod-rna/cod-rna", 59535, 8, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************ ijcnn1 ********************************************************** */
/* ****************************************************************************************************************** */

void load_ijcnn1(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/ijcnn1/ijcnn1", 49990, 22, X, y);
}

/* ****************************************************************************************************************** */
/* ************************************************* SUSY ********************************************************** */
/* ****************************************************************************************************************** */

void load_SUSY(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/SUSY/SUSY", 5000000, 18, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************ mnist ********************************************************* */
/* ****************************************************************************************************************** */

void load_mnist(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/mnist/mnist.scale", 60000, 784, X, y);

    /* transform into a binary problem: 0,1,2,3,4 -> -1; 5,6,7,8 -> +1 */
    y = (y.array() == 0 || y.array() == 1 || y.array() == 2 || y.array() == 3 || y.array() == 4).select(-1, y);
    y = (y.array() == 5 || y.array() == 6 || y.array() == 7 || y.array() == 8 || y.array() == 9).select(+1, y);

    assert((y.array() != -1 && y.array() != 1).sum() == 0);
}

/* ****************************************************************************************************************** */
/* ************************************************ mnist8m ********************************************************* */
/* ****************************************************************************************************************** */

void load_mnist8m(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/mnist8m/mnist8m.scale", 8100000, 784, X, y);

    /* transform into a binary problem: 0,1,2,3,4 -> -1; 5,6,7,8 -> +1 */
    y = (y.array() == 0 || y.array() == 1 || y.array() == 2 || y.array() == 3 || y.array() == 4).select(-1, y);
    y = (y.array() == 5 || y.array() == 6 || y.array() == 7 || y.array() == 8 || y.array() == 9).select(+1, y);

    assert((y.array() != -1 && y.array() != 1).sum() == 0);
}

/* ****************************************************************************************************************** */
/* *********************************************** gisette ********************************************************** */
/* ****************************************************************************************************************** */

void load_gisette(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* ignore the following features (see the corresponding README) */
    std::set<int> ignore_features = {
        111, 119, 196, 421, 479, 793, 876, 1037, 1040, 1179, 1267,
        1639, 1736, 1792, 1835, 1904, 2022, 2086, 2198, 2248, 2348,
        2585, 2604, 2686, 2691, 2811, 2901, 2909, 2952, 3007, 3026,
        3157, 3193, 3476, 3556, 3631, 3706, 3745, 3864, 4066, 4100,
        4255, 4388, 4872, 4964
    };

    /* read data */
    read_pascal_file("datasets/gisette/gisette_train.data", "datasets/gisette/gisette_train.labels", 6000, 5000, X, y, ignore_features);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* *********************************************** quantum ********************************************************** */
/* ****************************************************************************************************************** */

void load_quantum(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* set up number of samples and features */
    int N = 50000;
    int D = 65;

    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* read data */
    int dummy;
    FILE* file;
    file = fopen("datasets/quantum/phy_train.dat", "r");
    for (int i = 0; i < N; ++i) {
        /* ignore "EXAMPLE_ID" */
        dummy = fscanf(file, "%d", &dummy);

        /* read label */
        dummy = fscanf(file, "%d", &y(i));

        /* read features */
        double value;
        int j = 0;
        for (int j1 = 0; j1 < 78; ++j1) {
            /* read a number */
            dummy = fscanf(file, "%lf", &value);

            /* don't include the following features (see the corresponding README) */
            if (j1 >= 19 && j1 <= 21) continue;
            if (j1 >= 43 && j1 <= 45) continue;
            if (j1 == 28 || j1 == 54) continue;
            if (j1 >= 46 && j1 <= 50) continue;

            /* include the rest */
            X(i, j) = value;
            ++j;
        }
    }
    fclose(file);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);

    /* transform y from {0, 1} to {-1, 1} */
    y = 2 * y.array() - 1;
}

/* ****************************************************************************************************************** */
/* *********************************************** protein ********************************************************** */
/* ****************************************************************************************************************** */

void load_protein(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* set up number of samples and features */
    int N = 145751;
    int D = 74;

    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* read data */
    int dummy;
    FILE* file;
    file = fopen("datasets/protein/bio_train.dat", "r");
    for (int i = 0; i < N; ++i) {
        /* ignore "BLOCK_ID" and "EXAMPLE_ID" */
        dummy = fscanf(file, "%d", &dummy);
        dummy = fscanf(file, "%d", &dummy);

        /* read label */
        dummy = fscanf(file, "%d", &y(i));

        /* read features */
        for (int j = 0; j < D; ++j) {
            dummy = fscanf(file, "%lf", &X(i, j));
        }
    }
    fclose(file);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);

    /* transform y from {0, 1} to {-1, 1} */
    y = 2 * y.array() - 1;
}

/* ****************************************************************************************************************** */
/* ************************************************ alpha *********************************************************** */
/* ****************************************************************************************************************** */

void load_alpha(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/alpha/alpha_train.dat", "datasets/alpha/alpha_train.lab", 500000, 500, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************ epsilon ********************************************************* */
/* ****************************************************************************************************************** */

void load_epsilon(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/epsilon/epsilon_train.dat", "datasets/epsilon/epsilon_train.lab", 500000, 2000, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************** zeta ********************************************************** */
/* ****************************************************************************************************************** */

void load_zeta(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/zeta/zeta_train.dat", "datasets/zeta/zeta_train.lab", 500000, 2000, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************** beta ********************************************************** */
/* ****************************************************************************************************************** */

void load_beta(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/beta/beta_train.dat", "datasets/beta/beta_train.lab", 500000, 500, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************** gamma ********************************************************* */
/* ****************************************************************************************************************** */

void load_gamma(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/gamma/gamma_train.dat", "datasets/gamma/gamma_train.lab", 500000, 500, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* ************************************************** delta ********************************************************* */
/* ****************************************************************************************************************** */

void load_delta(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_pascal_file("datasets/delta/delta_train.dat", "datasets/delta/delta_train.lab", 500000, 500, X, y);

    /* scale features to [-1, 1] */
    scale_features(X, -1, 1);
}

/* ****************************************************************************************************************** */
/* *************************************************** fd *********************************************************** */
/* ****************************************************************************************************************** */

void load_fd(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/fd/fd_train.txt", 5469800, 900, X, y);

    /* scale features to [-1, 1] */
    //scale_features(X, -1, 1);
    assert((X.array() >= -1 && X.array() <= 1).all());
}

/* ****************************************************************************************************************** */
/* ************************************************** ocr *********************************************************** */
/* ****************************************************************************************************************** */

void load_ocr(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/ocr/ocr_train.txt", 3500000, 1156, X, y);

    /* scale features to [-1, 1] */
    //scale_features(X, -1, 1);
    assert((X.array() >= -1 && X.array() <= 1).all());
}

/* ****************************************************************************************************************** */
/* ************************************************ dna18 *********************************************************** */
/* ****************************************************************************************************************** */

void load_dna18(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/dna/dna_train.txt", 18000000, 800, X, y);

    /* scale features to [-1, 1] */
    //scale_features(X, -1, 1);
    assert((X.array() >= -1 && X.array() <= 1).all());
}
