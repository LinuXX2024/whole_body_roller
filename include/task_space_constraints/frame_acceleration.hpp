#pragma once

#include "constraint.hpp"
#include "dynamics.hpp"
#include <iostream>

namespace whole_body_roller {
    class FrameAccelerationConstraint : public whole_body_roller::ConstraintHandler {
    public:
        std::shared_ptr<whole_body_roller::Dynamics> dynamics;
        // std::shared_ptr<whole_body_roller::Constraint> constraint;
        std::string frame_name_;
        Eigen::VectorXd acceleration_target;

    public:
        FrameAccelerationConstraint(std::shared_ptr<whole_body_roller::Dynamics> dyn, 
                                    std::string frame_name) 
            : dynamics(dyn), frame_name_(frame_name) {
            
            this->acceleration_target = Eigen::VectorXd::Zero(6); // Initialize target acceleration to zero
            this->constraint = std::make_shared<whole_body_roller::Constraint>(
                this->dynamics->dec_v,
                6, // constraints for se3 acceleration
                whole_body_roller::constraint_type_t::EQUALITY // equality constraint for acceleration
            );
            this->constraint->set_tau_constraints(
                Eigen::MatrixXd::Zero(6, this->dynamics->dec_v->nv_ - 6) // No tau constraints for acceleration
            );
        }

        bool set_acceleration_target(const Eigen::VectorXd &acceleration) {
            if (acceleration.size() != 6) {
                return false; // size mismatch
            }
            this->acceleration_target = acceleration;
            return true;
        }

        bool update_constraint() override {
            std::cout << "updating frame acceleration constraint for frame: " << this->frame_name_ << "\n";
            if (!this->dynamics->is_dynamics_ready || !this->dynamics->model_->existFrame(this->frame_name_)) {
                return false; // dynamics not ready or frame does not exist
            }

            bool update_success = true;

            pinocchio::Data data_local(*this->dynamics->model_);
            pinocchio::computeJointJacobians(*this->dynamics->model_, 
                                            data_local, 
                                            this->dynamics->joint_positions_);

            pinocchio::updateFramePlacements(*this->dynamics->model_, data_local);          

            std::cout << "model->nv: " << this->dynamics->model_->nv << std::endl;
            std::cout << "dec_v->nv_: " << this->dynamics->dec_v->nv_ << std::endl;      
            pinocchio::FrameIndex fid = this->dynamics->model_->getFrameId(this->frame_name_);
            std::cout << "FrameId: " << fid << " / nframes = " << this->dynamics->model_->nframes << std::endl;
            std::cout << "model.nq = " << this->dynamics->model_->nq << ", model.nv = " << this->dynamics->model_->nv << std::endl;
            std::cout << "joint_positions_.size() = " << this->dynamics->joint_positions_.size() << std::endl;        
            std::cout << "model.nv = " << this->dynamics->model_->nv << std::endl;
            std::cout << "data_local.M.rows() = " << data_local.M.rows() << ", cols = " << data_local.M.cols() << std::endl;

            Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, this->dynamics->model_->nv);
            J.setZero();
            // Get the Jacobian of the frame
            pinocchio::getFrameJacobian(*this->dynamics->model_, 
                                        data_local, 
                                        this->dynamics->model_->getFrameId(this->frame_name_), 
                                        pinocchio::LOCAL_WORLD_ALIGNED,
                                        J);
                                        // see dynamics.cpp#111 for explanation as to why we use LOCAL_WORLD_ALIGNED

            Eigen::MatrixXd jacobian = J;

            //J = Eigen::MatrixXd::Zero(6, this->dynamics->model_->nv); for a test because the Jacobian contains very large numbers
            
            std::cout << "Jacobian: \n" << jacobian << std::endl;
        

            // std::cout << "jacobian size is " << jacobian.rows() << " x " << jacobian.cols() << "\n";
            std::cout << "computed jacobain for frame: " << this->frame_name_ << "\n";

            // Set the constraints for the acceleration
            update_success &= this->constraint->set_qdd_constraints(jacobian);
            std::cout << "set qdd constraints for frame: " << this->frame_name_ << " update success : " << update_success << "\n";

            double dt = 1e-6; // Small time step for numerical stability

            Eigen::VectorXd q_fut = Eigen::VectorXd::Zero(this->dynamics->model_->nq);
            q_fut = pinocchio::integrate(*this->dynamics->model_, 
                                         this->dynamics->joint_positions_, 
                                         this->dynamics->joint_velocities_ * dt);
             std::cout << "old q: \n" <<  this->dynamics->joint_positions_ << std::endl;                            
            std::cout << "q_fut: \n" << q_fut << std::endl;
            //pinocchio::Data data_local2(*this->dynamics->model_);

            pinocchio::computeJointJacobians(*this->dynamics->model_, 
                                            data_local, 
                                            q_fut);
            pinocchio::updateFramePlacements(*this->dynamics->model_, data_local);

            Eigen::Matrix<double, 6, Eigen::Dynamic> J_fut(6, this->dynamics->model_->nv);
            J_fut.setZero();

            pinocchio::getFrameJacobian(*this->dynamics->model_, 
                                        data_local, 
                                        this->dynamics->model_->getFrameId(this->frame_name_), 
                                        pinocchio::LOCAL_WORLD_ALIGNED, J_fut);

            Eigen::MatrixXd jacobian_fut = J_fut;

            Eigen::MatrixXd dJ = (jacobian_fut - jacobian) / dt; // Numerical derivative of the Jacobian
            std::cout << "dt: \n" << dt <<"\n";
            std::cout << "jac f: \n" << jacobian_fut <<"\n";
            std::cout << "jac : \n" << jacobian <<"\n";
            std::cout << "Matrix dJ: \n" << dJ <<"\n";
            Eigen::VectorXd res = this->acceleration_target - dJ * this->dynamics->joint_velocities_; 
            std::cout << "Resulting Matrix: \n" << res <<"\n";
            update_success &= this->constraint->set_constraint_bias(this->acceleration_target - dJ * this->dynamics->joint_velocities_); // Bias is the negative of the target acceleration
           
            // the selection matrix is set to all zeros in the constructor,
            // the contact force constraints are undefined. they all need to be zero, 
            // but they need to be defined again here since we don't know 
            // if the num contacts have remained the same since the last time this function was called
            this->constraint->ignore_contact_constraints();
            // this needs to be called everytime as it refreshes the thingi based on the number of contact points
            
             std::cout << "acceleration constraint for frame: " << this->frame_name_ << "updated" <<"\n";
            return update_success;
        } 
    };








    class BaseAccelerationConstraint : public whole_body_roller::ConstraintHandler {
    public:
        std::shared_ptr<whole_body_roller::Dynamics> dynamics;
        // std::shared_ptr<whole_body_roller::Constraint> constraint;
        //std::string frame_name_;
        Eigen::VectorXd acceleration_target;

    public:
        BaseAccelerationConstraint(std::shared_ptr<whole_body_roller::Dynamics> dyn ) 
            : dynamics(dyn){
            
            this->acceleration_target = Eigen::VectorXd::Zero(6); // Initialize target acceleration to zero
            this->constraint = std::make_shared<whole_body_roller::Constraint>(
                this->dynamics->dec_v,
                6, // constraints for se3 acceleration
                whole_body_roller::constraint_type_t::EQUALITY // equality constraint for acceleration
            );
            this->constraint->set_tau_constraints(
                Eigen::MatrixXd::Zero(6, this->dynamics->dec_v->nv_ - 6) // No tau constraints for acceleration
            );
        }

        bool set_acceleration_target(const Eigen::VectorXd &acceleration) {
            if (acceleration.size() != 6) {
                return false; // size mismatch
            }
            this->acceleration_target = acceleration;
            return true;
        }

        bool update_constraint() override {
            if (!this->dynamics->is_dynamics_ready) {
                std::cout << "base dynamics ready: " << this->dynamics->is_dynamics_ready <<"\n";
                return false; // dynamics not ready 
            }
            bool update_success = true;

            Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, this->dynamics->model_->nv);

            Eigen::MatrixXd selection_matrix = Eigen::MatrixXd::Zero(6, this->dynamics->model_->nv);

            Eigen::MatrixXd selection_matrix_floating_base = Eigen::MatrixXd::Identity(6,6);
            Eigen::MatrixXd selection_matrix_joints = Eigen::MatrixXd::Zero(6, this->dynamics->model_->nv - 6);
            selection_matrix << selection_matrix_floating_base, 
                                 selection_matrix_joints;

            // Set the constraints for the acceleration
            update_success &= this->constraint->set_qdd_constraints(selection_matrix);
            std::cout << "qdd set  \n" << selection_matrix <<"\n";

            update_success &= this->constraint->set_constraint_bias(this->acceleration_target); 
            std::cout << "bias set  \n" << this->acceleration_target <<"\n";

            this->constraint->ignore_contact_constraints();
            // this needs to be called everytime as it refreshes the thingi based on the number of contact points
            std::cout << "update_success \n" << update_success <<"\n";

            return update_success;
        } 
    };







    class ContactForceConstraint : public whole_body_roller::ConstraintHandler {
    public:
        std::shared_ptr<whole_body_roller::Dynamics> dynamics;
        double pi = 3.14159265358979323846;
        std::string frame_name_;
        //Eigen::VectorXd acceleration_target; // For bias

    public:
        ContactForceConstraint(std::shared_ptr<whole_body_roller::Dynamics> dyn , std::string frame_name ) 
            : dynamics(dyn), frame_name_(frame_name){
            
            //this->acceleration_target = Eigen::VectorXd::Zero(6); // Initialize target acceleration to zero

            this->constraint = std::make_shared<whole_body_roller::Constraint>(
                this->dynamics->dec_v,
                9, // constraints for se3 acceleration
                whole_body_roller::constraint_type_t::INEQUALITY // inequality equality constraint for forces
            );
            std::cout << "tau in contact force :  " << this->dynamics->dec_v->nv_ - 6<< "\n";

            this->constraint->set_tau_constraints(
                Eigen::MatrixXd::Zero(6, this->dynamics->dec_v->nv_ - 6) // No tau constraints for acceleration
            );
        }


        bool update_constraint() override {
            std::cout << "updateing COntact force constraint  " <<"\n";

            if (!this->dynamics->is_dynamics_ready) {
                return false; // dynamics not ready 
            }
            bool update_success = true;

           int k = 8;
           float mu = 0.6;
           
           std::vector<Eigen::MatrixXd> contact_;

           Eigen::VectorXd b = Eigen::VectorXd::Zero(k+1);
           Eigen::MatrixXd A = Eigen::MatrixXd::Zero(k+1, 3);
           Eigen::MatrixXd contact_matrix = Eigen::MatrixXd::Zero(k+1, 6);
           Eigen::MatrixXd mu_vec = (-mu)*(Eigen::MatrixXd::Ones(k+1, 1));
           Eigen::MatrixXd S(k+1, 2);
            std::cout << "vectores declared " <<"\n";

            for(int i = 0; i < k; i++){
                float theta = 2*pi*i/k + pi/k;
                Eigen::Vector2f s = {std::cos(theta) , std::sin(theta)} ; 
                S(i,0) = s[0];
                S(i,1) = s[1];
            }
            S(k,0) = 0.0f;
            S(k,1) = 0.0f;
            mu_vec(k) = -1.0f;

            std::cout << "after S "<< S <<"\n";
            A << S, mu_vec;
            std::cout << "after A "<< A <<"\n";

            Eigen::MatrixXd filler = Eigen::MatrixXd::Zero(k+1, 3);
            contact_matrix << A ,filler;
            
            Eigen::MatrixXd second_contact_dummy =  Eigen::MatrixXd::Zero(k+1, 6);

            if(frame_name_ == "right_foot"){
                contact_.push_back(contact_matrix);
                contact_.push_back(second_contact_dummy);
            }else if(frame_name_ == "left_foot"){
                contact_.push_back(second_contact_dummy);
                contact_.push_back(contact_matrix);
            }

            std::cout << "after push "<< contact_matrix <<"\n";            
            Eigen::MatrixXd zero_qdd = (Eigen::MatrixXd::Zero(k+1, this->dynamics->dec_v->nv_));

            update_success &= this->constraint->set_qdd_constraints(zero_qdd);
            this->constraint->set_tau_constraints(
                Eigen::MatrixXd::Zero(k+1, this->dynamics->dec_v->nv_ - 6) // No tau constraints for acceleration
            );
            std::cout << "tau c set "<< update_success <<"\n";

            update_success &= this->constraint->set_constraint_bias(b); 
   
            this->constraint->contacts_are_considered = true;
            this->constraint->set_contact_constraints(contact_);
            std::cout << "update succses " << update_success <<"\n";

            return update_success;
        } 
    };
}