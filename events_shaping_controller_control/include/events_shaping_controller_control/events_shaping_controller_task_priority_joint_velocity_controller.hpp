#pragma once

#include <mutex>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka_msgs/SetJointImpedance.h>

#include <geometry_msgs/TwistStamped.h>

#include <Eigen/Dense>

#include <franka_hw/franka_model_interface.h>

namespace events_shaping_controller_control {

class TPJointVelocityController : public controller_interface::MultiInterfaceController<
                                               franka_hw::FrankaModelInterface,
                                               hardware_interface::VelocityJointInterface,
                                               franka_hw::FrankaStateInterface>
{
  public:
    // ros_control standard functions
    bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
    void update(const ros::Time&, const ros::Duration& period) override;
    void starting(const ros::Time&) override;
    void stopping(const ros::Time&) override;

    const double MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL{0.05}; // s
    const double MAX_TIME_FOR_NON_REALTIME_THREAD{0.025}; // s

  private:
    // Non-real-time optimization loop. This will make sure main control loop
    // is not bogged down by arbitrarily long time-consuming tasks
    ros::Timer task_priority_optimizer_timer_;
    void taskPriorityOptimizerCallback(const ros::TimerEvent& timer_obj);
    ros::Time last_time_nonrt_thread_done_;

    // Variables to exchange control commands between non-RT and RT threads
    std::array<double, 6> curr_command_camera_frame_;
    Eigen::Vector<double, 7> curr_command_joint_vels_eigen_;

    // Callbacks and helpers for processing input commands from the velocity commanders
    void gazeLockCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);
    void heuristicCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);
    void approachCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);
    ros::Time last_rcvd_time_gaze_lock_commands_;
    ros::Time last_rcvd_time_heuristic_commands_;
    ros::Time last_rcvd_time_approach_commands_;
    std::mutex gaze_lock_commands_mutex_;
    std::mutex heuristic_commands_mutex_;
    std::mutex approach_commands_mutex_;    
    ros::Subscriber sub_gaze_lock_commands_;
    ros::Subscriber sub_heuristic_commands_;
    ros::Subscriber sub_approach_commands_;
    double requested_vX_;
    double requested_vY_;
    double requested_vZ_;
    double requested_wX_;
    double requested_wY_;
    double requested_wZ_;

    // A simple "smoother" that enforces max acceleration
    void smoothTrajectory(const Eigen::Vector<double, 7>& desired_values,
                          Eigen::Vector<double, 7>& smoothed_values_r);

    // Some handlers to talk to FCI
    hardware_interface::VelocityJointInterface* velocity_joint_interface_;
    std::vector<hardware_interface::JointHandle> velocity_joint_handles_;
    std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
    std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
    std::string arm_id_;

    // Some helpers for getting transform to camera
    void getInitTransformTimerCallback(const ros::TimerEvent& timer_event);
    ros::Timer init_transform_timer_;
    bool init_timer_ran_out_;
    Eigen::Affine3d F_T_C_eigen;
    std::array<double, 16> F_T_C_array;
    bool F_T_C_available_;

    /////////////////////////////////////////////////////////////////////////////////////////
    // GENERIC task priority joint velocity control
    // This can be used in other realizations too, and not just for GCVS
    template<int TASK_DIM>
    void computeMaintaskJointVels(
      const Eigen::Matrix<double, TASK_DIM, 7>& main_task_jacobian,
      const Eigen::Vector<double, TASK_DIM>& main_task_commands,
      Eigen::Vector<double, 7>& main_task_computed_joint_vels_r,
      Eigen::Matrix<double, 7, TASK_DIM>& main_task_jacobian_pseudoinverse_r
    );

    // NOTE: Also stores computes pseudo inverse of this_subtask_jacobian 
    // This will be useful for saving compute while chaining
    template<int PREV_TASK_DIM, int TASK_DIM>
    void computeSubtaskJointVels(
      const Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_jacobian,
      const Eigen::Vector<double, TASK_DIM>& this_subtask_commands,
      const Eigen::Matrix<double, PREV_TASK_DIM, 7>& prev_subtask_jacobian,
      const Eigen::Matrix<double, 7, PREV_TASK_DIM>& prev_subtask_jacobian_pseudoinverse,
      const Eigen::Vector<double, 7>& prev_subtask_joint_vels,
      const Eigen::Matrix<double, 7, 7>& prev_subtask_nullspace_projector,
      Eigen::Vector<double, 7>& this_subtask_computed_joint_vels_r,
      Eigen::Matrix<double, 7, 7>& this_subtask_nullspace_projector_r,
      Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_augmented_jacobian_r,
      Eigen::Matrix<double, 7, TASK_DIM>& this_subtask_augmented_jacobian_pseudoinverse_r
    );

    template<int PREV_TASK_DIM, int TASK_DIM>
    void computeFullRankSubtaskJointVels(
      const Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_jacobian,
      const Eigen::Vector<double, TASK_DIM>& this_subtask_commands,
      const Eigen::Matrix<double, PREV_TASK_DIM, 7>& prev_subtask_jacobian,
      const Eigen::Matrix<double, 7, PREV_TASK_DIM>& prev_subtask_jacobian_pseudoinverse,
      const Eigen::Vector<double, 7>& prev_subtask_joint_vels,
      const Eigen::Matrix<double, 7, 7>& prev_subtask_nullspace_projector,
      Eigen::Vector<double, 7>& this_subtask_computed_joint_vels_r,
      Eigen::Matrix<double, 7, 7>& this_subtask_nullspace_projector_r,
      Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_augmented_jacobian_r,
      Eigen::Matrix<double, 7, TASK_DIM>& this_subtask_augmented_jacobian_pseudoinverse_r
    );

    template<int PREV_TASK_DIM, int TASK_DIM>
    void computeDynamicallyConsistentSubtaskJointVels(
      const Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_jacobian,
      const Eigen::Vector<double, TASK_DIM>& this_subtask_commands,
      const Eigen::Matrix<double, PREV_TASK_DIM, 7>& prev_subtask_jacobian,
      const Eigen::Matrix<double, 7, PREV_TASK_DIM>& prev_subtask_jacobian_pseudoinverse,
      const Eigen::Vector<double, 7>& prev_subtask_joint_vels,
      const Eigen::Matrix<double, 7, 7>& prev_subtask_nullspace_projector,
      Eigen::Vector<double, 7>& this_subtask_computed_joint_vels_r,
      Eigen::Matrix<double, 7, 7>& this_subtask_nullspace_projector_r,
      Eigen::Matrix<double, TASK_DIM, 7>& this_subtask_augmented_jacobian_r,
      Eigen::Matrix<double, 7, TASK_DIM>& this_subtask_augmented_jacobian_pseudoinverse_r
    );

    // This is a Moore-Penrose if jacobian_jacobian_T is full rank
    // Else, this will result in BAD numbers for pseudoinverse 
    template<int TASK_DIM>
    void vanillaJacobianPseudoInverse(
      const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
      Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
    );

    template<int TASK_DIM>
    void moorePenrosePseudoInverse(
      const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
      Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
    );

    // Weighted pseudo inverse, to ensure dynamic consistency
    // The weight matrix is the inertia matrix
    template<int TASK_DIM>
    void dynamicallyConsistentJacobianPseudoInverse(
        const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
        Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
    );
    /////////////////////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////////////////////////
    void computeManipulatorHessianFromManipulatorJacobian(
      const Eigen::Matrix<double, 6, 7>& robot_jacobian,
      std::array<Eigen::Matrix<double, 6, 7>, 7>& robot_hessian_r);
    /////////////////////////////////////////////////////////////////////////////////////////

    // Panda joint limits
    // It is possible to get this from URDF, but the process
    // is long and arduous and processing it is potentially error-prone
    const std::array<const std::array<double, 2>, 7> PANDA_JOINT_LIMITS_ = 
    {
      {
        // {-q_lim (rad), q_lim (rad)}
        {-2.8973, 2.8973},
        {-1.7628, 1.7628},
        {-2.8973, 2.8973},
        {-3.0718, -0.0698},
        {-2.8973, 2.8973},
        {-0.0175, 3.7525},
        {-2.8973, 2.8973}
      }
    };

    const double JOINT_LIMIT_POTENTIAL_GAIN_ = 1e0;
    const double JOINT_LIMIT_INFLUENCE_DISTANCE_ = 0.15; // rad

    // const double MANIPULABILITY_TASK_GAIN_ = 1e-3;
};

} // namespace events_shaping_controller