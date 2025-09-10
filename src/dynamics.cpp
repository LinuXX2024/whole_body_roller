#include "constraint.hpp"
#include "dynamics.hpp"
#include <iostream>

namespace whole_body_roller {
    Dynamics::Dynamics(int num_end_effectors, std::shared_ptr<pinocchio::Model> model) {
        this->num_end_effectors_ = num_end_effectors;
        this->model_ = model;
        this->data_ = std::make_shared<pinocchio::Data>(*this->model_);
        std::cout << "Initialized dynamics with model having " << this->model_->nq << " dof and " << this->model_->nv << " velocity variables\n";

        this->dec_v = std::make_shared<whole_body_roller::ControlDecisionVariables>(
            this->model_->nv, // number of variables in the second derivative of the joint positions (incl. floating base)
            0 // number of contact points, will be updated later based on the end effectors in contact
        );


        this->dynamics_constraint = std::make_shared<whole_body_roller::Constraint>(
            this->dec_v,
            this->model_->nv, // number of variables in the second derivative of the joint positions (incl. floating base)
            whole_body_roller::constraint_type_t::EQUALITY // the dynamics constraint is an equality constraint
        );

        this->constraint = this->dynamics_constraint; // set the constraint to the dynamics constraint
        this->dynamics_constraint->ignore_contact_constraints(false); // ignore contact constraints by default

        Eigen::MatrixXd selection_matrix = Eigen::MatrixXd::Zero(this->model_->nv, this->model_->nv - 6);
        Eigen::MatrixXd selection_matrix_floating_base = Eigen::MatrixXd::Zero(6, this->model_->nv - 6);
        Eigen::MatrixXd selection_matrix_joints = Eigen::MatrixXd::Identity(this->model_->nv - 6, this->model_->nv - 6);
        selection_matrix << selection_matrix_floating_base, 
                            selection_matrix_joints;
        this->dynamics_constraint->set_tau_constraints(
            (-1)*(selection_matrix)
        );
        // M(q)qdd - S tau - J_c lambda_c = -h(q, qd)
        // that is why the selection matrix is multiplied by -1

        // all end effectors need to be added
        this->added_end_effectors = 0;
        this->is_dynamics_ready = false;
        this->free_end_effectors = num_end_effectors;


    }


    bool Dynamics::add_end_effector(std::string frame_name) {
        if (this->added_end_effectors < this->num_end_effectors_ &&
            this->model_->existFrame(frame_name)
        ) {
            whole_body_roller::EndEffector ee;
            ee.frame = frame_name;
            ee.state = whole_body_roller::end_effector_state_t::FLOATING; // default state
            ee.fn = whole_body_roller::end_effector_function_t::IDLE; // default function
            this->end_effectors.push_back(ee);
            this->end_effector_map_[frame_name] = this->added_end_effectors;
            this->added_end_effectors++;

            std::cout << "added end effector " << this->added_end_effectors << " out of " << this->num_end_effectors_ << "\n";

            if (this->added_end_effectors == this->num_end_effectors_) {
                this->is_dynamics_ready = true;
            }
            return true;
        }
        std::cout << "the frame does not exist \n";
        return false;
    }

    bool Dynamics::change_end_effector_state(std::string frame_name, end_effector_state_t new_state) {
        if (this->model_->existFrame(frame_name) &&
            this->end_effector_map_.count(frame_name) == 1
            ) {
            this->end_effectors[this->end_effector_map_[frame_name]].state = new_state;
            std::cout << "updating nc after changing end effector state \n";
            this->update_nc();
            return true;
        }

        return false;
    }

    bool Dynamics::change_end_effector_function(std::string frame_name, end_effector_function_t new_function) {
        if (this->model_->existFrame(frame_name) &&
            this->end_effector_map_.count(frame_name) == 1
            ) {
            this->end_effectors[this->end_effector_map_[frame_name]].fn = new_function;    
            return true;
        }

        return false;
    }

    bool Dynamics::update_joint_states(const Eigen::VectorXd &joint_positions, const Eigen::VectorXd &joint_velocities) {
        if (joint_positions.size() != this->model_->nq || joint_velocities.size() != this->model_->nv) {
            return false; // size mismatch
        }

        this->joint_positions_ = joint_positions;
        this->joint_velocities_ = joint_velocities;

        // Update the kinematics for the model
        // pinocchio::forwardKinematics(*this->model_, *this->data_, this->joint_positions_, this->joint_velocities_);
        
        // Update the dynamics constraint with the new joint states
        // this->dynamics_constraint->update_dynamics_constraint(this->joint_positions_, this->joint_velocities_);

        return true;
    }

    void Dynamics::update_nc() {
        int nc = 0;
        for (const auto& ee : this->end_effectors) {
            if (ee.state == whole_body_roller::end_effector_state_t::IN_CONTACT) {
                nc++;
            }
        }
        std::cout << "the number of contacts are " << nc << "\n";
        this->dec_v->nc_ = nc;
    }
        
    bool Dynamics::update_dynamics_constraint() {
        if (!this->is_dynamics_ready) {
            std::cout << "The dynamics is not ready \n";
            return false; // dynamics not ready
        }

        std::vector<Eigen::MatrixXd> contact_jacobians;

        bool update_success = true;

        pinocchio::forwardKinematics(*this->model_, *this->data_, this->joint_positions_, this->joint_velocities_);
        pinocchio::computeJointJacobians(*this->model_, *this->data_, this->joint_positions_);
        // Update the number of contact points in the decision variables
        for (const auto& ee : this->end_effectors) {
            if (ee.state == whole_body_roller::end_effector_state_t::IN_CONTACT) {
                /*
                contact_jacobians.push_back(
                    // We use LOCAL_WORLD_ALIGNED to get the jacobian in the world frame
                    // it basically gives us the jacobian of the end effector in the world frame
                    // if we were to think abou tit in terms of the acceleration or velocity,
                    //      it gives us the velocity of the frame we want to 
                    //      consider with respect to the world frame
                    (-1) * (pinocchio::getFrameJacobian(*this->model_, 
                                                *this->data_, 
                                                this->model_->getFrameId(ee.frame), 
                                                pinocchio::LOCAL_WORLD_ALIGNED).transpose())
                );
                // M(q)qdd - S tau - J_c lambda_c = -h(q, qd)
                // that is why the jacobian is multiplied by -1
                */
                std::cout << "im here upadating a daynamics constraint\n";

                Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, this->model_->nv);
                J.setZero();
                // 2) fill J (this overload writes into J) — if your pinocchio version supports it:
                pinocchio::getFrameJacobian(*this->model_, *this->data_,
                                            this->model_->getFrameId(ee.frame),
                                            pinocchio::LOCAL_WORLD_ALIGNED,
                                            J); // J is now (6 x nv)
                // 3) compute the desired matrix (-J).transpose() and force evaluation
                Eigen::MatrixXd mat = (-J).transpose().eval(); // now a fully materialized (nv x 6) matrix
                std::cout << "mat: \n" <<  mat << std::endl;
                // 4) push the evaluated matrix into the vector
                contact_jacobians.push_back(std::move(mat));
            }
        }
        update_success &= this->dynamics_constraint->set_contact_constraints(contact_jacobians);

        // need to figure out which convention to use see: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html#ab48efbd527d1bc9941da1a5f400e751a
        // this is to be left to the default, which is Convention::LOCAL
        // why? see https://chatgpt.com/share/686f2db8-4f7c-800e-86e3-665b6a965391

        pinocchio::crba(*this->model_, *this->data_, this->joint_positions_); // Calculates Mass matrix M(q)
        std::cout << "M for qdd constraints: \n" << this->data_->M << std::endl;
        update_success &= this->dynamics_constraint->set_qdd_constraints(this->data_->M);

        std::cout << "model: \n" << *this->model_<< std::endl;
        std::cout << "data: \n" << this->data_ << std::endl;
        std::cout << "model positions: \n" << this->joint_positions_<< std::endl;
        std::cout << "model velocities \n" << this->joint_velocities_<< std::endl;

        pinocchio::rnea(*this->model_, 
                        *this->data_, 
                        this->joint_positions_, 
                        this->joint_velocities_, 
                        Eigen::VectorXd::Zero(this->model_->nv)); // Calculates non linear Terms h(q, dq)
        std::cout << "Tau that is added to dyn const " << this->data_->tau << std::endl;
        update_success &= this->dynamics_constraint->set_constraint_bias((-1)*(this->data_->tau)); // -h(q, qd)
        // we set the acceleration of rnea to 0 to avoid it computing the inverse dynamics of the Mass matrix
        // that is being done separately using crba as it needs to be fed into a different constraint

        std::cout << "Dynamics updated successfully \n"; 
        return update_success;
    }

    std::shared_ptr<whole_body_roller::ControlDecisionVariables> Dynamics::get_dec_v() {
        return this->dec_v;
    }

    bool Dynamics::update_constraint() {
        return this->update_dynamics_constraint();
    }
 
}