#pragma once

#include <mutex>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_msgs/SetJointImpedance.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <geometry_msgs/TwistStamped.h>

#include <Eigen/Dense>

namespace events_shaping_controller_control {

class CartesianVelocityController : public controller_interface::MultiInterfaceController<
                                               franka_hw::FrankaVelocityCartesianInterface,
                                               franka_hw::FrankaStateInterface>
{
  public:
    bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
    void update(const ros::Time&, const ros::Duration& period) override;
    void starting(const ros::Time&) override;
    void stopping(const ros::Time&) override;

    const double MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL{0.01}; // s

  private:
    void gazeLockCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);
    void heuristicCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);
    void approachCommandCallback(const geometry_msgs::TwistStampedConstPtr& msg);

    void saturateVelocities(std::array<double, 6> &vels);

    franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_;
    std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;
    // std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;

    bool homogeneous_transform_to_vel_transform(
      const Eigen::Affine3d &pose_transform, 
      Eigen::Matrix<double, 6, 6>& vel_transform,
      bool do_full_transform);

    void getInitTransformTimerCallback(const ros::TimerEvent& timer_event);
    ros::Timer init_transform_timer_;
    bool init_timer_ran_out_;

    std::string arm_id_;

    ros::Subscriber sub_gaze_lock_commands_;
    ros::Subscriber sub_heuristic_commands_;
    ros::Subscriber sub_approach_commands_;

    ros::Time last_rcvd_time_gaze_lock_commands_;
    ros::Time last_rcvd_time_heuristic_commands_;
    ros::Time last_rcvd_time_approach_commands_;

    std::mutex gaze_lock_commands_mutex_;
    std::mutex heuristic_commands_mutex_;
    std::mutex approach_commands_mutex_;

    double requested_vX_;
    double requested_vY_;
    double requested_vZ_;
    double requested_wX_;
    double requested_wY_;
    double requested_wZ_;

    Eigen::Affine3d EE_T_C_eigen;
    bool EE_T_C_available_;
    std::array<double, 6> curr_command_camera_frame_;
};

} // namespace events_shaping_controller