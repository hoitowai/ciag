# ciag

MATLAB and C++ implementation for the paper "On Curvature-aided Incremental Aggregated Gradient Methods" by H.-T. Wai, W. Shi, C. A. Uribe, A. Nedic, A. Scaglione (online: https://arxiv.org/abs/1806.00125)

The C++ programs are developed from the code base in https://github.com/arodomanov/nim_icml16_code, written by Anton Rodomanov based on the paper "A Superlinearly-Convergent Proximal Newton-type Method for the Optimization of Finite Sums", in Proc. of ICML, July, 2016.

# Usage of the MATLAB programs

For synthetic data only. The main program can be found in "main_sim.m". The parameters for different algorithms are specified within the source code.

<b>Warning</b>: Be reminded that most of the implementation have not been fully optimized.

# Usage of the C++ programs

For A-CIAG and CIAG, The setting "kappa" (default = 0.0001) controls the scaling of the step size "gamma" in the paper such that the step size used is "kappa / L", where "L" is the smoothness constant. The A-CIAG uses an extrapolation rate controlled by "beta" (default = 0.99), note that it was called "alpha" in the paper.

The dataset are not included on github. They have to be downloaded independently through the "download.sh" script in their respective folders.

Example 1: executing
- ./main --method CIAG --dataset a9a --max_epochs 100 --minibatch_size 5

runs the CIAG method on the dataset a9a with a minibatch size of 5, with a default step size of 0.0001 / L.

Example 2: executing
- ./main --method ACIAG --dataset a9a --max_epochs 100 --minibatch_size 5 --kappa 0.00015 --beta 0.95

runs the A-CIAG method on the dataset a9a with a minibatch size of 5, with a step size of 0.00015 / L and extrapolation rate of 0.95.

The program has been compiled and tested on a computer running MacOS X 10.13 with gcc 4.2.1.

<b>Warning</b>: The authors have only tested the implementation for the methods CIAG, A-CIAG, NIM and SAG. Be reminded that the implementation have not been fully optimized.
