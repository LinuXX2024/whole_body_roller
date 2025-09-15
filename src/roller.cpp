#include "roller.hpp"
#include <iostream>
#include <casadi_helpers.hpp>

namespace whole_body_roller {
    // this function is just to construct the roller object
    // this function also adds the dynamics constraint to the roller which is to be there by default.
    // The dynamics constraint is the one that ensures that
    //      the joint accelerations are consistent with the dynamics model.
    // The dynamics object is also the only constraint that is allowed to modify dec_v
    Roller::Roller(std::shared_ptr<whole_body_roller::Dynamics> dyn) {

        // this->dec_v = dv;
        this->dynamics = dyn;
        this->dec_v = this->dynamics->get_dec_v();
        this->add_constraint(this->dynamics->dynamics_constraint, this->dynamics);
    }

    // this function is to add constraints to the controller.
    // The constraints could be on the contact forces, torques, 
    // or the joint accelerations (or the task space by using the jacobian to transform \ddot{q})
    bool Roller::add_constraint(std::shared_ptr<whole_body_roller::Constraint> constraint, std::shared_ptr<whole_body_roller::ConstraintHandler> constraint_handler) {
        if (constraint->dec_v->nv_ == this->dec_v->nv_) {
            this->constraints.push_back(constraint);
            // we can add the constraint handler to the constraint handler list
            // so that we can update the constraints before solving the qp
            this->constraint_handlers.push_back(constraint_handler);
            std::cout << "adding constraint with " << constraint->num_constraints_ << " constraints\n";
            this->consolidate_constraints(); // update the number of equality and inequality constraints
            this->update_optim(); // update the qp with the new number of constraints
            return true;
        }
        return false; // size mismatch
    }
    // this function consolidates all the constraints, coutns the number of equality and inequality constraints,
    void Roller::consolidate_constraints() {
        this->num_eq_constraints = 0;
        this->num_ineq_constraints = 0;
        // TODO implement this function to consolidate all the constraints
        for (auto& constraint_handler : this->constraint_handlers) {
            // if (!constraint_handler->constraint_is_active()) {
            //     continue; // skip the constraint if it is not active
            // } // we still need to count the inactive constraints to maintain the size of the constraint matrices and zero them out
            auto constraint = constraint_handler->constraint; // get the constraint from the handler
            // if (!constraint->is_constraint_valid()) {
            //     return false; // if any constraint is not valid, we return false
            // }
            if (constraint->constraint_type_ == whole_body_roller::constraint_type_t::EQUALITY) {
                num_eq_constraints += constraint->num_constraints_;
            } else if (constraint->constraint_type_ == whole_body_roller::constraint_type_t::INEQUALITY) {
                num_ineq_constraints += constraint->num_constraints_;
            }
        }
    }


    // adds equality and ineq constraints, to the qp and then solves them
    bool Roller::solve_qp() {
        // int num_eq_constraints = 0;
        // int num_ineq_constraints = 0;
        int nvars = this->dec_v->get_ndv(); // number of variables in the qp
        std::cout << "number of decision variables: " << nvars << "\n";
        std::cout << "size of qdd " << this->dec_v->nv_  << ", and the size of tau is " << this->dec_v->ntau_ << ", and the size of the contacts is " << this->dec_v->nc_ << "\n";
        // Create equality the constraint matrix
        Eigen::MatrixXd eq_constraint_matrix(nvars, num_eq_constraints);
        eq_constraint_matrix.setZero();
        Eigen::VectorXd eq_constraint_bias(num_eq_constraints);
        eq_constraint_bias.setZero();
        Eigen::MatrixXd ineq_constraint_matrix(nvars, num_ineq_constraints);
        ineq_constraint_matrix.setZero();
        Eigen::VectorXd ineq_constraint_bias(num_ineq_constraints);
        ineq_constraint_bias.setZero();


        // iterate through all constraints and append the constraint matrices and biases
        int eq_con_col = 0;
        int ineq_con_col = 0;
        for (auto& constraint_handler : this->constraint_handlers) {
            auto constraint = constraint_handler->constraint; // get the constraint from the handler
            // if (!constraint_handler->constraint_is_active()) {
            //     continue; // skip the constraint if it is not active
            // } // we still need to add the inactive constraints to maintain the size of the constraint matrices and zero them out
            if (constraint->constraint_type_ == whole_body_roller::constraint_type_t::EQUALITY) {
                // append the equality constraint matrix and bias
                if (constraint_handler->constraint_is_active()) {
                    eq_constraint_matrix.block(0, eq_con_col, nvars, constraint->num_constraints_) = 
                        constraint->get_constraint_matrix().transpose();
                    eq_constraint_bias.segment(eq_con_col, constraint->num_constraints_) = 
                        constraint->constraint_bias;
                } else {
                    eq_constraint_matrix.block(0, eq_con_col, nvars, constraint->num_constraints_) = 
                        Eigen::MatrixXd::Zero(nvars, constraint->num_constraints_);
                    eq_constraint_bias.segment(eq_con_col, constraint->num_constraints_) = 
                        Eigen::VectorXd::Zero(constraint->num_constraints_);
                }
                std::cout << " constraint activ? : \n" << constraint_handler->is_constraint_active << std::endl;
                std::cout << " not finished constraint bias: \n" << eq_constraint_bias << std::endl;
                eq_con_col += constraint->num_constraints_;
            } else if (constraint->constraint_type_ == whole_body_roller::constraint_type_t::INEQUALITY) {
                // append the inequality constraint matrix and bias
                if (constraint_handler->constraint_is_active()) {
                    ineq_constraint_matrix.block(0, ineq_con_col, nvars, constraint->num_constraints_) = 
                        constraint->get_constraint_matrix().transpose();
                    ineq_constraint_bias.segment(ineq_con_col, constraint->num_constraints_) = 
                        constraint->constraint_bias;
                } else {
                    ineq_constraint_matrix.block(0, ineq_con_col, nvars, constraint->num_constraints_) = 
                        Eigen::MatrixXd::Zero(nvars, constraint->num_constraints_);
                    ineq_constraint_bias.segment(ineq_con_col, constraint->num_constraints_) = 
                        Eigen::VectorXd::Zero(constraint->num_constraints_);
                }
                
                
                ineq_con_col += constraint->num_constraints_;
            }
        }
        std::cout << "added all constraints \n";

        std::cout << "trying to set casadi params\n";
        if (this->num_eq_constraints > 0) {
            std::cout << "eq constraints exist, adding them\n";
            std::cout << "shape of eq_con " << this->optim->eq_con_param_.size() << std::endl;
            std::cout << "shape of eigen mat " << eq_constraint_matrix.rows() << ", " << eq_constraint_matrix.cols() << "\n";

            Eigen::MatrixXd tmp = eq_constraint_matrix.transpose().eval();
            casadi::DM dm_eqm = casadi_helpers::toDM(tmp);

            std::cout << "running set value\n";
            // this->optim->opti.set_value(this->optim->test_param, casadi_helpers::toDMcol(Eigen::VectorXd::Ones(1))); // this works or does it?
            // std::cout << "1d parameter value set\n";
            std::cout << "shape of param " << this->optim->eq_con_param_.size() << std::endl;
            // std::cout << "the eq matrix is " << eq_constraint_matrix << std::endl;
            // std::cout << "shape of dm" << dm_eqm.size() << std::endl;
            this->optim->opti.set_value(this->optim->eq_con_param_, dm_eqm); 
            std::cout << "eq constraints exist, added matrix, now adding bias\n";
            Eigen::VectorXd tmp_bias = eq_constraint_bias.eval();
            this->optim->opti.set_value(this->optim->eq_bias_param_, casadi_helpers::toDMcol(tmp_bias));
        }

        if (this->num_ineq_constraints > 0) {
            std::cout << "ineq constraints exist, adding them\n";
            this->optim->opti.set_value(this->optim->ineq_con, casadi_helpers::toDM(ineq_constraint_matrix.transpose()));
            this->optim->opti.set_value(this->optim->ineq_bias, casadi_helpers::toDMcol(ineq_constraint_bias));
        }
        
        //set slack variables
        Eigen::VectorXd slack_vector = Eigen::VectorXd(this->num_eq_constraints); 
        slack_vector.setOnes();         
        slack_vector.tail(this->num_eq_constraints - 6).setConstant(0.0); 

        this->optim->opti.set_initial(this->optim->slack_variables, casadi_helpers::toDMcol(slack_vector));
        std::cout << "slac variables "<<  this->optim->slack_variables << std::endl;


        std::cout << "set casadi params\n";
        std::cout << "bias size:\n" << eq_constraint_bias.size() << std::endl;
        std::cout << "eq_bias :\n" << eq_constraint_bias << std::endl;
        std::cout << "eq_constraint_matrix :\n" << eq_constraint_matrix.transpose() << std::endl;
        std::cout << "ieq_bias :\n" << ineq_constraint_bias << std::endl;
        std::cout << "ineq_constraint_matrix :\n" << ineq_constraint_matrix.transpose() << std::endl;
        //std::cout << "optim bias :\n" << this->optim->eq_con_param_ << std::endl;
        //std::cout << "optim constraint_matrix :\n" << eq_constraint_matrix.transpose().eval() << std::endl;
        try {
            auto sol = this->optim->opti.solve();
            std::cout << "QP solved" << std::endl;

            casadi::DM sol_z = sol.value(this->optim->z);
            Eigen::VectorXd sol_z_eigen = casadi_helpers::DM_to_Vector(sol_z);
            std::cout << "QP solved with \n" << sol_z_eigen << std::endl;
            
            std::cout << "nv_-1 \n" << this->dec_v->nv_-1 << std::endl;
            std::cout << "ntau \n" << this->dec_v->ntau_ << std::endl;

            // std::cout << "the solution is " << sol_z_eigen << std::endl;
            // std::cout << "the size of the solution is "  << sol_z_eigen.size() << "\n";
            this->joint_torques = std::make_shared<Eigen::VectorXd>(sol_z_eigen.segment(this->dec_v->nv_, this->dec_v->ntau_));
            std::cout << "the torques are " << *(this->joint_torques) << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Solver failed with exception: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    void Roller::update_optim() {
        // DONE implement this function to update the optim object
        // std::cout << "updating optim with decv " << this->dec_v->nv_ << ", and the size of tau is " << this->dec_v->ntau_ << ", and the size of the contacts is " << this->dec_v->nc_ << "\n";
        this->optim = std::make_shared<casadi_helpers::CaSolver>(
            this->dec_v->get_ndv(), 
            this->num_eq_constraints, 
            this->num_ineq_constraints);
    }


    // TODO we need to provide an interface to the constraint handlers
    // we need to be able to call update_joint_states() on the dynamics
    // and we need to be able to call set_acceleration_target() on the frame acceleration constraints
   /*
    * This funciton is the main control loop, it will update the dynamics 
           and the parameters of the constraints based on dynamics->joint_positions_ and dynamics->joint_velocities_
    */
    bool Roller::step() {
        // first update the dynamics constraint as that changes dec_v
        // then update the frame/joint acceleration constraints
        std::cout << "checking if constraint is valid\n" << this->constraint_handlers.size()<<" \n";
        for (auto& handler : this->constraint_handlers) {
            if (!handler->update_constraint()) {
                std::cout << "checking if constraint is valid\n";
                return false; // if any constraint fails to update, we return false
            }
        }

        // TODO check if dec_v->get_ndvs() has changed or if the number of constraints 
        // have changed and update the optimizer if it has
        // are there any other cases where we need to update the optimizer?
        std::cout << "befor consolidating \n";
        this->consolidate_constraints();
        if (this->optim->z.size().first != this->dec_v->get_ndv() || 
            this->num_eq_constraints != this->optim->eq_con.size().first || 
            this->num_ineq_constraints != this->optim->ineq_con.size().first
        ) {
            std::cout << "need to rebuild optim\n";
            this->update_optim();
        }

        std::cout << "updated all constraints \n";

        std::cout << "steeeee" << "e\n" << "e\n" << "e\n" << "e\n" << "e\n" << "e\n" << "e\n" << "e\n" << "ep taken\n"<< std::endl;
        return this->solve_qp(); // solve the qp with the updated constraints
    }


}
