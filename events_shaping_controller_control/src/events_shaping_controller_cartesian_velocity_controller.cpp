
#include <events_shaping_controller_control/events_shaping_controller_cartesian_velocity_controller.hpp>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

namespace events_shaping_controller_control {

bool CartesianVelocityController::init(hardware_interface::RobotHW* robot_hardware,
                                            ros::NodeHandle& node_handle)
{
    // Most of the init copied from franka_ros example_controllers
    if (!node_handle.getParam("arm_id", arm_id_)) 
    {
        ROS_ERROR("CartesianVelocityController: Could not get parameter arm_id");
        return false;
    }

    velocity_cartesian_interface_ =
        robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
    if (velocity_cartesian_interface_ == nullptr) 
    {
        ROS_ERROR(
            "CartesianVelocityController: Could not get Cartesian velocity interface from "
            "hardware");
        return false;
    }

    try
    {
        velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
            velocity_cartesian_interface_->getHandle((arm_id_ + "_robot")));
    } 
    catch (const hardware_interface::HardwareInterfaceException& e) 
    {
        ROS_ERROR_STREAM(
            "CartesianVelocityController: Exception getting Cartesian handle: " << e.what());
        return false;
    }

    // Set appropriate joint impedances
    std::vector<double> impedances;
    if(!node_handle.getParam("joint_internal_controller_impedances", impedances))
    {
        ROS_WARN("CartesianVelocityController: Could not get parameter joint_internal_controller_impedances. Will use defaults.");
        impedances = {1000., 1000., 1000., 1000., 1000., 1000., 1000.};
    }
    else if (impedances.size() != 7)
    {
        ROS_WARN_STREAM(
            "CartesianVelocityController: Invalid size " << impedances.size() << " for joint_internal_controller_impedances."
            << " Will use defaults.");
        impedances = {1000., 1000., 1000., 1000., 1000., 1000., 1000.};
    }


    ros::ServiceClient service_client_ = node_handle.serviceClient<
                                            franka_msgs::SetJointImpedance>("/franka_control/set_joint_impedance");

    franka_msgs::SetJointImpedance srv;
    std::copy(impedances.begin(), impedances.end(), srv.request.joint_stiffness.begin());
    if(!service_client_.waitForExistence(ros::Duration(4.0)))
    {
        ROS_WARN("CartesianVelocityController: /franka_control/set_joint_impedance service not available.");
    }

    if(service_client_.call(srv))
    {
        if (!((bool)srv.response.success))
        {
            ROS_WARN("CartesianVelocityController: Failed to set joint impedance values. Robot will continue using previous values.");
        }
    }
    else
    {
        ROS_WARN("CartesianVelocityController: Could not call service to set joint impedances. Robot will continue using previous values.");
    }

    ///////////////////////////////////////////////////////////////////////////////////
    // Below gets state in joints. But I only need Cartesian pose
    // Retained for posterity
    // auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
    // if (state_interface == nullptr) 
    // {
    //     ROS_ERROR("CartesianVelocityController: Could not get state interface from hardware");
    //     return false;
    // }
    // try
    // {
    //     state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
    //         state_interface->getHandle(arm_id_ + "_robot"));
    // }
    // catch(hardware_interface::HardwareInterfaceException& e)
    // {
    //     ROS_ERROR_STREAM(
    //        "CartesianVelocityController: Exception getting state handle: " 
    //        << e.what());
    //     return false;
    // }
    ///////////////////////////////////////////////////////////////////////////////////

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&CartesianVelocityController::gazeLockCommandCallback, this, _1);
        subscribe_options.init("/gaze_lock_commander/wXY", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_gaze_lock_commands_ = node_handle.subscribe(subscribe_options);
    }

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&CartesianVelocityController::heuristicCommandCallback, this, _1);
        subscribe_options.init("/heuristic_commander/vXYandwZ", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_heuristic_commands_ = node_handle.subscribe(subscribe_options);
    }

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&CartesianVelocityController::approachCommandCallback, this, _1);
        subscribe_options.init("/approach_commander/vZ", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_approach_commands_ = node_handle.subscribe(subscribe_options);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: Getting transform here to EE does not work reliably
    // That's because _EE is published by franka_control which waits for the controllers
    // to be loaded! (Facepalm!) So there'll be deadlock. So it'll have to be unfortunately moved
    // to real-time parts of the code. Retaining below for posterity.
    //
    // tf::StampedTransform transform;
    // tf::TransformListener listener;
    // try
    // {
    //     std::string camera_frame = "camera_color_optical_frame";
    //     if (listener.waitForTransform(arm_id_ + "_EE", camera_frame, ros::Time(0),
    //                                 ros::Duration(10.0))) 
    //     {
    //         listener.lookupTransform(arm_id_ + "_EE", camera_frame, ros::Time(0),
    //                                 transform);
    //     } 
    //     else 
    //     {
    //     ROS_ERROR_STREAM(
    //         "CartesianVelocityController: Failed to read transform from " 
    //         << arm_id_ + "_EE" << " to " << camera_frame);
    //     return false;
    //     }
    // } 
    // catch (tf::TransformException& ex) 
    // {
    //     ROS_ERROR_STREAM(
    //         "CartesianVelocityController: " << ex.what());
    //     return false;
    // }
    // tf::transformTFToEigen(transform, EE_T_C_eigen);    
    ///////////////////////////////////////////////////////////////////////////////////////////////

    EE_T_C_available_ = false;
    {
        boost::function<void(const ros::TimerEvent&)> callback = 
                boost::bind(&CartesianVelocityController::getInitTransformTimerCallback, this, _1);
        init_transform_timer_ = node_handle.createTimer(ros::Duration(0.01), callback, true);
        init_timer_ran_out_ = false;
    }

    return true;
}

void CartesianVelocityController::getInitTransformTimerCallback(const ros::TimerEvent& timer_event)
{
    tf::StampedTransform transform;
    tf::TransformListener listener;
    try
    {
        std::string camera_frame = "event_camera_optical_frame";
        if (listener.waitForTransform(arm_id_ + "_EE", camera_frame, ros::Time(0),
                                    ros::Duration(10.0))) 
        {
            listener.lookupTransform(arm_id_ + "_EE", camera_frame, ros::Time(0),
                                    transform);
            tf::transformTFToEigen(transform, EE_T_C_eigen); 
            EE_T_C_available_ = true;
        } 
        else 
        {
        ROS_ERROR_STREAM(
            "CartesianVelocityController: Failed to read transform from " 
            << arm_id_ + "_EE" << " to " << camera_frame);
        }
    } 
    catch (tf::TransformException& ex) 
    {
        ROS_ERROR_STREAM(
            "CartesianVelocityController: " << ex.what());
    }
    init_timer_ran_out_ = true;
}

void CartesianVelocityController::starting(const ros::Time& /* time */) 
{
    curr_command_camera_frame_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
}

void CartesianVelocityController::stopping(const ros::Time& /*time*/) 
{
    // Following NOTE from franka_ros example_controllers
    // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
    // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
    // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

void CartesianVelocityController::update(const ros::Time& curr_time,
                                            const ros::Duration& period) 
{
    // Setup EE to C transform. And if not available, don't send any control commands! (not even zero)
    if (!EE_T_C_available_)
    {
        std::string camera_frame = "event_camera_optical_frame";
        if (!init_timer_ran_out_)
        {
            ROS_WARN_STREAM_THROTTLE(1.0,
            "CartesianVelocityController: Transform between " 
            << arm_id_ + "_EE" << " and " << camera_frame 
            << " unavailable. Robot WON'T be controlled until it is available!");
        }
        else
        {
            ROS_ERROR_STREAM_ONCE(
            "CartesianVelocityController: Transform between " 
            << arm_id_ + "_EE" << " and " << camera_frame 
            << " NOT FOUND. Robot WON'T be controlled!");
        }
        return;
    }

    double gaze_lock_commands_time_diff = MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL;
    {
        std::lock_guard<std::mutex> _(gaze_lock_commands_mutex_);
        gaze_lock_commands_time_diff = (curr_time - last_rcvd_time_gaze_lock_commands_).toSec();
        if(gaze_lock_commands_time_diff > MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Gaze control commands not received for " << gaze_lock_commands_time_diff << " s"
            );

            // Gently slow down instead of suddenly stopping
            curr_command_camera_frame_[3] *= 0.998;
            curr_command_camera_frame_[4] *= 0.998;
        }
        else
        {
            curr_command_camera_frame_[3] = requested_wX_;
            curr_command_camera_frame_[4] = requested_wY_;
        }
    }

    {
        std::lock_guard<std::mutex> _(heuristic_commands_mutex_);
        const double time_diff = (curr_time - last_rcvd_time_heuristic_commands_).toSec();
        if(time_diff > MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Heuristic commands not received for " << time_diff << " s"
            );

            // Gently slow down instead of suddenly stopping
            curr_command_camera_frame_[0] *= 0.998;
            curr_command_camera_frame_[1] *= 0.998;
            curr_command_camera_frame_[2] *= 0.998;
            curr_command_camera_frame_[5] *= 0.998;
        }
        else
        {
            curr_command_camera_frame_[0] = requested_vX_;
            curr_command_camera_frame_[1] = requested_vY_;
            curr_command_camera_frame_[2] = requested_vZ_;
            curr_command_camera_frame_[5] = requested_wZ_;
        }
    }

    {
        std::lock_guard<std::mutex> _(approach_commands_mutex_);
        const double time_diff = (curr_time - last_rcvd_time_approach_commands_).toSec();
        if(time_diff > MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Approach commands not received for " << time_diff << " s"
            );

            // Gently slow down instead of suddenly stopping
            curr_command_camera_frame_[2] *= 0.998;
        }
        // Additional safety check for approach commands to avoid colliding into objects
        // that are not even gaze locked on: either because the object was moved out of sight
        // or because object came up too close (and hence no features on object detected)
        // else if(gaze_lock_commands_time_diff > MAX_TIME_FOR_REFUSING_REQUESTED_COMMAND_VEL)
        // {
        //    ROS_WARN_STREAM_THROTTLE(
        //        2.0,
        //        "Approach commands are not being honored because no gaze lock commands received in " << gaze_lock_commands_time_diff << " s"
        //    );

            // Gently slow down instead of suddenly stopping
        //    curr_command_camera_frame_[2] *= 0.998;
        //}
        //else
        //{
        //    curr_command_camera_frame_[2] = requested_vZ_;
        //}
        curr_command_camera_frame_[2] = requested_vZ_;
    }

    auto O_T_EE = velocity_cartesian_handle_->getRobotState().O_T_EE;
    Eigen::Affine3d O_T_EE_eigen(Eigen::Matrix4d::Map(O_T_EE.data()));

    Eigen::Matrix<double, 6, 6> O_P_EE_eigen;
    Eigen::Matrix<double, 6, 6> EE_P_C_eigen;

    homogeneous_transform_to_vel_transform(O_T_EE_eigen, O_P_EE_eigen, false);
    homogeneous_transform_to_vel_transform(EE_T_C_eigen, EE_P_C_eigen, true);

    Eigen::Matrix<double, 6, 1> curr_command_camera_frame_eigen = Eigen::Map<
                        Eigen::Matrix<double, 6, 1>>(curr_command_camera_frame_.data());

    Eigen::Matrix<double, 6, 1> curr_command_base_frame_eigen = 
                        O_P_EE_eigen * EE_P_C_eigen * curr_command_camera_frame_eigen;

    std::array<double, 6> curr_command_base_frame;
    Eigen::Map<Eigen::Matrix<double, 6, 1>>(&curr_command_base_frame[0], 6, 1) = curr_command_base_frame_eigen.matrix();

    // std::cout << "Curr. comm. base frame = " << curr_command_base_frame_eigen << std::endl;

    saturateVelocities(curr_command_base_frame);
    velocity_cartesian_handle_->setCommand(curr_command_base_frame);
}

void CartesianVelocityController::saturateVelocities(std::array<double, 6> &vels)
{
    // Limits given according to specifications
    // const double p_t_max = 1.7 / 1000; // 1.7 m / s --> m / ms
    const double p_r_max = 2.5 / 1000; // 2.5 rad / s --> rad / ms

    // Scaled down limits
    const double p_t_max = 1.0 / 1000; // 1.0 m / s --> m / ms
    // const double p_r_max = 2.0 / 1000; // 2.0 rad / s --> rad / ms

    auto vels_d = velocity_cartesian_handle_->getRobotState().O_dP_EE_d;

    for(size_t i = 0; i < 6; i++)
    {
        const double max_lim = i <= 2 ? p_t_max : p_r_max;
        const double difference = vels[i] - vels_d[i];
        vels[i] = vels_d[i] + std::max(std::min(difference, max_lim), -max_lim);
    }
}

void CartesianVelocityController::gazeLockCommandCallback(
    const geometry_msgs::TwistStampedConstPtr& msg)
{
    {
        std::lock_guard<std::mutex> _(gaze_lock_commands_mutex_);
        requested_wX_ = msg->twist.angular.x;
        requested_wY_ = msg->twist.angular.y;
        last_rcvd_time_gaze_lock_commands_ = msg->header.stamp;
    }
    // ROS_INFO
}

void CartesianVelocityController::heuristicCommandCallback(
    const geometry_msgs::TwistStampedConstPtr& msg)
{
    {
        std::lock_guard<std::mutex> _(heuristic_commands_mutex_);
        requested_vX_ = msg->twist.linear.x;
        requested_vY_ = msg->twist.linear.y;
        requested_vZ_ = msg->twist.linear.z;
        requested_wZ_ = msg->twist.angular.z;
        last_rcvd_time_heuristic_commands_ = msg->header.stamp;
    }
}

void CartesianVelocityController::approachCommandCallback(
    const geometry_msgs::TwistStampedConstPtr& msg)
{
    {
        std::lock_guard<std::mutex> _(approach_commands_mutex_);
        requested_vZ_ = msg->twist.linear.z;
        last_rcvd_time_approach_commands_ = msg->header.stamp;
    }
}

bool CartesianVelocityController::homogeneous_transform_to_vel_transform(
    const Eigen::Affine3d &pose_transform,
    Eigen::Matrix<double, 6, 6>& vel_transform,
    bool do_full_transform)
{
    auto rot = pose_transform.linear();
    auto tra = pose_transform.translation();

    vel_transform = Eigen::Matrix<double, 6, 6>::Zero();

    // Only rotation-based
    if (!do_full_transform)
    {
        vel_transform.block<3, 3>(0, 0) = rot;
        vel_transform.block<3, 3>(3, 3) = rot;
    }
    // Full transform
    else
    {
        vel_transform.block<3, 3>(0, 0) = rot;
        Eigen::Matrix3d skew_sym_tra;
        skew_sym_tra << 0, -tra(2), tra(1),
                        tra(2), 0, -tra(0),
                        -tra(1), tra(0), 0;
        vel_transform.block<3, 3>(0, 3) = skew_sym_tra * rot;
        vel_transform.block<3, 3>(3, 3) = rot;
    }

    return true;
}


}  // namespace events_shaping_controller_control

PLUGINLIB_EXPORT_CLASS(events_shaping_controller_control::CartesianVelocityController,
                       controller_interface::ControllerBase)