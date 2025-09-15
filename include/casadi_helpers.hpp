#pragma once

#include <casadi/casadi.hpp>
// #include <Eigen/Core>
#include <Eigen/Dense>


namespace casadi_helpers {
    // 1) Dense (MatrixXd or any dense expression) -> DM
    template <class Derived>
    casadi::DM toDM(const Eigen::MatrixBase<Derived>& M) {
    // Make a concrete, contiguous, column-major copy (eval() materializes expressions/views)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C = M.eval();
    casadi::DM dm = casadi::DM::zeros(C.rows(), C.cols());
    // std::cout << "shape of basee mat " << C.rows() << ", " << C.cols() << "\n";
    std::memcpy(dm.ptr(), C.data(), sizeof(double)*C.rows()*C.cols());
    // std::cout << "done memcpy\n";
    return dm;
    }

    // 2) Sparse -> DM (via dense)
    inline casadi::DM toDM(const Eigen::SparseMatrix<double>& S) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C = Eigen::MatrixXd(S);
    casadi::DM dm = casadi::DM::zeros(C.rows(), C.cols());
    // std::cout << "shape of sparse mat " << C.rows() << ", " << C.cols() << "\n";
    std::memcpy(dm.ptr(), C.data(), sizeof(double)*C.rows()*C.cols());
    return dm;
    }

    // 3) Vector convenience (ensures n×1 shape)
    inline casadi::DM toDMcol(const Eigen::VectorXd& v) {
    casadi::DM dm = casadi::DM::zeros(v.size(), 1);
    std::memcpy(dm.ptr(), v.data(), sizeof(double)*v.size());
    return dm;
    }


    // DM -> Eigen::MatrixXd (handles dense or sparse-pattern DM)
    inline Eigen::MatrixXd DM_to_Eigen(const casadi::DM& M) {
        const casadi::Sparsity& S = M.sparsity();
        const int m = S.size1(), n = S.size2();
        Eigen::MatrixXd E = Eigen::MatrixXd::Zero(m, n);

        if (S.is_dense()) {
            if (m*n) std::memcpy(E.data(), M.ptr(), sizeof(double)*m*n); // col-major → col-major
            return E;
        }

        return E;
    }

    // DM vector (n×1 or 1×n) -> Eigen::VectorXd
    inline Eigen::VectorXd DM_to_Vector(const casadi::DM& M) {
        const int m = M.size1(), n = M.size2();
        if (!(m == 1 || n == 1))
            throw std::runtime_error("DM_to_Vector: input is not a vector (expected n×1 or 1×n).");

        const casadi::Sparsity& S = M.sparsity();

        Eigen::VectorXd v = Eigen::VectorXd::Zero(m);
        if (m) std::memcpy(v.data(), M.ptr(), sizeof(double)*m);
        return v;
    }    

    class CaSolver {
    public:
        using ParamT = decltype(std::declval<casadi::Opti>().parameter(0,0));
        ParamT eq_con_param_;
        ParamT eq_bias_param_;
        casadi::Opti opti;
        casadi::MX z; // Decision variables
        casadi::MX slack_variables; // slack variables
        casadi::MX obj; // Objective function
        casadi::MX eq_con; // Equality constraints
        casadi::MX eq_bias; // Equality constraint bias
        casadi::MX ineq_con; // Inequality constraints
        casadi::MX ineq_bias; // Inequality constraint bias
        // casadi::MX test_param;


        CaSolver(int ndv, int neq, int nineq) {
            // this->opti = casadi::Opti();
            // this->test_param = this->opti.parameter();
            this->z = this->opti.variable(ndv); // decision variables
            this->slack_variables = this->opti.variable(neq); // slack variables

            int nv_ = 25;
            int ntau_ = 19;
            int n_fc = 12; //should not be hardcoded

            Eigen::VectorXd weights(ndv);
            weights.setOnes();         
            weights.tail(n_fc).setConstant(0.01); // less penalization for contact forces
            weights.segment(nv_, ntau_).setConstant(5.0); // more penelization for torque
            std::vector<double> w_vec(weights.data(), weights.data() + weights.size());
            std::cout << "weights "<<  w_vec << std::endl;

            casadi::MX W = casadi::MX::diag(casadi::DM(w_vec));
            casadi::MX Wslack = 1000 * casadi::MX::eye(neq); //Weights for slack variable


            if (neq > 0) {
                this->eq_con_param_ = this->opti.parameter(neq, ndv);
                this->eq_bias_param_ = this->opti.parameter(neq);
                this->eq_con = this->eq_con_param_;
                this->eq_bias = this->eq_bias_param_;
                this->opti.subject_to(casadi::MX::mtimes(eq_con, z) + this->slack_variables == eq_bias); // equality constraints

                this->opti.subject_to(this->slack_variables >= 0); //slack constraint
            }
            if (nineq > 0) {
                this->ineq_con = this->opti.parameter(nineq, ndv);
                this->ineq_bias = this->opti.parameter(nineq);
                this->opti.subject_to(casadi::MX::mtimes(ineq_con, z) <= ineq_bias); // inequality constraints
            }
            this->obj = casadi::MX::mtimes({z.T(), W, z})  + casadi::MX::mtimes({this->slack_variables.T(), Wslack, this->slack_variables}); // Initialize objective function to zero
            this->opti.minimize(this->obj); // Set the objective function
            this->opti.solver("ipopt");
        }
    };
}