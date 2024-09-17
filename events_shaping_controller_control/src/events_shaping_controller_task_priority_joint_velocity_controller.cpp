
#include <events_shaping_controller_control/events_shaping_controller_task_priority_joint_velocity_controller.hpp>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

namespace events_shaping_controller_control {

bool TPJointVelocityController::init(hardware_interface::RobotHW* robot_hardware,
                                            ros::NodeHandle& node_handle)
{
    // Lot of init copied from franka_ros example_controllers
    if (!node_handle.getParam("arm_id", arm_id_)) 
    {
        ROS_ERROR("TPJointVelocityController: Could not get parameter arm_id");
        return false;
    }

    velocity_joint_interface_ = 
        robot_hardware->get<hardware_interface::VelocityJointInterface>();
    if (velocity_joint_interface_ == nullptr) 
    {
        ROS_ERROR(
            "TPJointVelocityController: Could not get joint velocity interface from "
            "hardware");
        return false;
    }

    std::vector<std::string> joint_names;
    if (!node_handle.getParam("joint_names", joint_names))
    {
        ROS_ERROR("TPJointVelocityController: Could not parse joint names. "
                  "Please provide joint names as ROS params");
        return false;
    }
    if (joint_names.size() != 7)
    {
        ROS_ERROR_STREAM("TPJointVelocityController: Incompatible number of joints! " << 
                         "This controller only supports 7 joints. " << 
                         "Received: " << joint_names.size() << " joints!");
        return false;
    }

    velocity_joint_handles_.resize(7);
    for (size_t i = 0; i < 7; ++i)
    {
        try
        {
            velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
        }
        catch(const hardware_interface::HardwareInterfaceException& ex)
        {
            ROS_ERROR_STREAM(
                "TPJointVelocityController: Exception getting joint handles: " << ex.what());
            return false;
        }
        
    }

    auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
    if (state_interface == nullptr)
    {
        ROS_ERROR("TPJointVelocityController: Could not get state interface from hardware");
        return false;
    }
    try
    {
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
            state_interface->getHandle(arm_id_ + "_robot"));
    }
    catch(const hardware_interface::HardwareInterfaceException& ex)
    {
        ROS_ERROR_STREAM(
            "TPJointVelocityController: Exception getting state handle from interface: " 
            << ex.what());
        return false;
    }

    auto* model_interface = robot_hardware->get<franka_hw::FrankaModelInterface>();
    if (model_interface == nullptr) {
        ROS_ERROR_STREAM(
            "TPJointVelocityController: Error getting model interface from hardware");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle(arm_id_ + "_model"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "TPJointVelocityController: Exception getting model handle from interface: "
            << ex.what());
        return false;
    }
    

    // Set appropriate joint impedances
    std::vector<double> impedances;
    if(!node_handle.getParam("joint_internal_controller_impedances", impedances))
    {
        ROS_WARN("TPJointVelocityController: Could not get parameter joint_internal_controller_impedances. Will use defaults.");
        impedances = {1000., 1000., 1000., 1000., 1000., 1000., 1000.};
    }
    else if (impedances.size() != 7)
    {
        ROS_WARN_STREAM(
            "TPJointVelocityController: Invalid size " << impedances.size() << " for joint_internal_controller_impedances."
            << " Will use defaults.");
        impedances = {1000., 1000., 1000., 1000., 1000., 1000., 1000.};
    }

    ros::ServiceClient service_client_ = node_handle.serviceClient<
                                            franka_msgs::SetJointImpedance>("/franka_control/set_joint_impedance");

    franka_msgs::SetJointImpedance srv;
    std::copy(impedances.begin(), impedances.end(), srv.request.joint_stiffness.begin());
    if(!service_client_.waitForExistence(ros::Duration(4.0)))
    {
        ROS_WARN(
            "TPJointVelocityController: /franka_control/set_joint_impedance service not available.");
    }

    if(service_client_.call(srv))
    {
        if (!((bool)srv.response.success))
        {
            ROS_WARN(
                "TPJointVelocityController: Failed to set joint impedance values. Robot will continue using previous values.");
        }
    }
    else
    {
        ROS_WARN(
            "TPJointVelocityController: Could not call service to set joint impedances. Robot will continue using previous values.");
    }

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&TPJointVelocityController::gazeLockCommandCallback, this, _1);
        subscribe_options.init("/gaze_lock_commander/wXY", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_gaze_lock_commands_ = node_handle.subscribe(subscribe_options);
    }

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&TPJointVelocityController::heuristicCommandCallback, this, _1);
        subscribe_options.init("/heuristic_commander/vXYandwZ", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_heuristic_commands_ = node_handle.subscribe(subscribe_options);
    }

    {
        ros::SubscribeOptions subscribe_options;

        boost::function<void(const geometry_msgs::TwistStampedConstPtr&)> callback = 
            boost::bind(&TPJointVelocityController::approachCommandCallback, this, _1);
        subscribe_options.init("/approach_commander/vZ", 1, callback);

        subscribe_options.transport_hints = ros::TransportHints().reliable().tcpNoDelay();

        sub_approach_commands_ = node_handle.subscribe(subscribe_options);
    }

    F_T_C_available_ = false;
    {
        boost::function<void(const ros::TimerEvent&)> callback = 
                boost::bind(&TPJointVelocityController::getInitTransformTimerCallback, this, _1);
        init_transform_timer_ = node_handle.createTimer(ros::Duration(0.01), callback, true);
        init_timer_ran_out_ = false;
    }

    {
        boost::function<void(const ros::TimerEvent&)> callback = 
                boost::bind(&TPJointVelocityController::taskPriorityOptimizerCallback, this, _1);
        
        // NOTE this rate of 1000 Hz is NOT guaranteed. This is non-real-time and in fact setup
        // separate from the main real-time control thread in case the time taken to compute
        // task-priority-based joint velocities takes more than 1 ms.
        task_priority_optimizer_timer_ = node_handle.createTimer(ros::Duration(1./1000.), callback);
        // task_priority_optimizer_timer_ = node_handle.createTimer(ros::Duration(0.1), callback);
    }

    return true;
}

void TPJointVelocityController::getInitTransformTimerCallback(
    const ros::TimerEvent& timer_event)
{
    tf::StampedTransform transform;
    tf::TransformListener listener;
    try
    {
        std::string camera_frame = "event_camera_optical_frame";
        if (listener.waitForTransform(arm_id_ + "_link8", camera_frame, ros::Time(0),
                                    ros::Duration(10.0))) 
        {
            listener.lookupTransform(arm_id_ + "_link8", camera_frame, ros::Time(0),
                                    transform);
            tf::transformTFToEigen(transform, F_T_C_eigen);
            std::copy_n(F_T_C_eigen.data(), 16, F_T_C_array.begin());
            F_T_C_available_ = true;
        } 
        else 
        {
        ROS_ERROR_STREAM(
            "TPJointVelocityController: Failed to read transform from " 
            << arm_id_ + "_link8" << " to " << camera_frame);
        }
    } 
    catch (tf::TransformException& ex) 
    {
        ROS_ERROR_STREAM(
            "TPJointVelocityController: " << ex.what());
    }
    init_timer_ran_out_ = true;
}

void TPJointVelocityController::starting(const ros::Time& /* time */) 
{
    // Panda doesn't allow switching to a different controller
    // when the robot's under movement. The built-in stopping
    // behavior kicks in. And so switching to this controller
    // will have the semantic zero of zero velocity.
    curr_command_camera_frame_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
}

void TPJointVelocityController::stopping(const ros::Time& /*time*/) 
{
    // Following NOTE from franka_ros example_controllers
    // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
    // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
    // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

void TPJointVelocityController::update(const ros::Time& curr_time,
                                     const ros::Duration& period) 
{
    // Setup Flange to Camera transform. 
    // And if not available, don't send any control commands! (not even zero)
    if (!F_T_C_available_)
    {
        std::string camera_frame = "event_camera_optical_frame";
        if (!init_timer_ran_out_)
        {
            ROS_WARN_STREAM_THROTTLE(1.0,
            "TPJointVelocityController: Transform between " 
            << arm_id_ + "_link8" << " and " << camera_frame 
            << " unavailable. Robot WON'T be controlled until it is available!");
        }
        else
        {
            ROS_ERROR_STREAM_ONCE(
            "TPJointVelocityController: Transform between " 
            << arm_id_ + "_link8" << " and " << camera_frame 
            << " NOT FOUND. Robot WON'T be controlled!");
        }
        return;
    }
    else
    {
        // No reason to run the check for transform once it's received.
        // Following does nothing if the timer is already stopped.
        init_transform_timer_.stop();
    }

    const double rt_to_nonrt_time_diff = (curr_time - last_time_nonrt_thread_done_).toSec();
    if (rt_to_nonrt_time_diff > MAX_TIME_FOR_NON_REALTIME_THREAD)
    {
        ROS_WARN_STREAM_THROTTLE(
                2.0,
                "Time slip between non-RT and RT thread is " << rt_to_nonrt_time_diff << " s!"
            );
        curr_command_joint_vels_eigen_ *= 0.998;
    }

    // TODO move this to rosparam
    const bool DO_SMOOTH_TRAJECTORY = true;

    // Finally set the calculated joint velocities
    if (DO_SMOOTH_TRAJECTORY)
    {
        Eigen::Vector<double, 7> smoothed_joint_vels_eigen;
        smoothTrajectory(curr_command_joint_vels_eigen_, smoothed_joint_vels_eigen);

        velocity_joint_handles_[0].setCommand(smoothed_joint_vels_eigen[0]);
        velocity_joint_handles_[1].setCommand(smoothed_joint_vels_eigen[1]);
        velocity_joint_handles_[2].setCommand(smoothed_joint_vels_eigen[2]);
        velocity_joint_handles_[3].setCommand(smoothed_joint_vels_eigen[3]);
        velocity_joint_handles_[4].setCommand(smoothed_joint_vels_eigen[4]);
        velocity_joint_handles_[5].setCommand(smoothed_joint_vels_eigen[5]);
        velocity_joint_handles_[6].setCommand(smoothed_joint_vels_eigen[6]);
    }
    else
    {
        velocity_joint_handles_[0].setCommand(curr_command_joint_vels_eigen_[0]);
        velocity_joint_handles_[1].setCommand(curr_command_joint_vels_eigen_[1]);
        velocity_joint_handles_[2].setCommand(curr_command_joint_vels_eigen_[2]);
        velocity_joint_handles_[3].setCommand(curr_command_joint_vels_eigen_[3]);
        velocity_joint_handles_[4].setCommand(curr_command_joint_vels_eigen_[4]);
        velocity_joint_handles_[5].setCommand(curr_command_joint_vels_eigen_[5]);
        velocity_joint_handles_[6].setCommand(curr_command_joint_vels_eigen_[6]);
    }

}

void TPJointVelocityController::taskPriorityOptimizerCallback(const ros::TimerEvent& timer_obj)
{
    if(!F_T_C_available_)
    {
        return;
    }

    const ros::Time curr_time = timer_obj.current_real;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Gather all requests
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
            curr_command_camera_frame_[5] *= 0.998;
        }
        else
        {
            curr_command_camera_frame_[0] = requested_vX_;
            curr_command_camera_frame_[1] = requested_vY_;
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
        //     ROS_WARN_STREAM_THROTTLE(
        //        2.0,
        //        "Approach commands are not being honored because no gaze lock commands received in " << gaze_lock_commands_time_diff << " s"
        //    );

        //    // Gently slow down instead of suddenly stopping
        //    curr_command_camera_frame_[2] *= 0.998;
        //}
        //else
        //{
        //    curr_command_camera_frame_[2] = requested_vZ_;
        //}
        curr_command_camera_frame_[2] = requested_vZ_;
    }
    Eigen::Matrix<double, 6, 1> curr_command_camera_frame_eigen = Eigen::Map<
                        Eigen::Matrix<double, 6, 1>>(curr_command_camera_frame_.data());
    ///////////////////////////////////////////////////////////////////////////////////////////////////////


    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // Find the joint velocities from requested commands
    franka::RobotState robot_state = state_handle_->getRobotState();
    
    std::array<double, 16> dummy_EE_T_K; // To satisfy getBodyJacobian

    // NOTE: Frame::kEndEffector is used as a proxy to camera frame
    // So Jacobian will be obtained with respect to camera directly
    // instead of the end effector. Also dummy_EE_T_K does not matter
    // as K is ahead of EE in kinematic chain and 
    // getBodyJacobian internally only gets Jacobian to the new "EE"
    std::array<double, 42> robot_jacobian_array = model_handle_->getBodyJacobian(
        franka::Frame::kEndEffector, robot_state.q, F_T_C_array, dummy_EE_T_K
    );
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> robot_jacobian(robot_jacobian_array.data());

    // ////////////////////////////////////////////////////////////////////////
    // Sanity check without tasks, but direct control of all DOFs
    // Eigen::Matrix<double, 6, 6> robot_jacobian_robot_jacobian_T =
    //                             robot_jacobian * robot_jacobian.transpose();
    // Eigen::Matrix<double, 7, 6> robot_jacobian_least_squares_pseudo_inverse = 
    //                             robot_jacobian.transpose() * 
    //                                 robot_jacobian_robot_jacobian_T.inverse();
    // auto combined_joint_vels = 
    //     robot_jacobian_least_squares_pseudo_inverse * curr_command_camera_frame_eigen;
    // // std::cout << combined_joint_vels << std::endl << std::endl;
    // ////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////
    // Task priority 1 -- Joint limit avoidance
    Eigen::Vector<double, 1> joint_limit_avoidance_command;
    joint_limit_avoidance_command[0] = -JOINT_LIMIT_POTENTIAL_GAIN_;
    Eigen::Matrix<double, 1, 7> joint_limit_avoidance_jacobian;
    joint_limit_avoidance_jacobian.setZero();

    for (int j = 0; j < 7; j++)
    {
        const double dist_to_upper_joint_limit = 
            PANDA_JOINT_LIMITS_[j][1] - robot_state.q[j];
        const double dist_to_lower_joint_limit = 
            robot_state.q[j] - PANDA_JOINT_LIMITS_[j][0];

        // Potential function is a log barrier, log(x)
        // So its Jacobian is simply 1/x
        if (dist_to_upper_joint_limit < dist_to_lower_joint_limit)
        {
            if (dist_to_upper_joint_limit <= JOINT_LIMIT_INFLUENCE_DISTANCE_)
            {
                joint_limit_avoidance_jacobian(0, j) += 
                    1 / dist_to_upper_joint_limit;

            }
            //else zero
        }
        else
        {
            if (dist_to_lower_joint_limit <= JOINT_LIMIT_INFLUENCE_DISTANCE_)
            {
                joint_limit_avoidance_jacobian(0, j) += 
                    -1 / dist_to_lower_joint_limit;
            }
            //else zero
        }
    }

    Eigen::Vector<double, 7> joint_vels_joint_limit_avoidance;
    const Eigen::Matrix<double, 7, 7> nullspace_projector_joint_limit_avoidance = 
        Eigen::Matrix<double, 7, 7>::Identity();  // First task has all of the "nullspace" available;
    const Eigen::Matrix<double, 1, 7> joint_limit_avoidance_augmented_jacobian = joint_limit_avoidance_jacobian;
    Eigen::Matrix<double, 7, 1> joint_limit_avoidance_augmented_jacobian_pseudoinverse;
    
    computeMaintaskJointVels<1>(
        joint_limit_avoidance_augmented_jacobian,
        joint_limit_avoidance_command,
        joint_vels_joint_limit_avoidance,
        joint_limit_avoidance_augmented_jacobian_pseudoinverse
    );

    // DEBUG
    // const ros::Time clock_after_joint_limit_task = ros::Time::now();
    // const double time_taken_after_joint_limit_task = (clock_after_joint_limit_task - curr_time).toSec();
    // std::cout << "time_taken_after_joint_limit_task = " << time_taken_after_joint_limit_task << std::endl;
    // std::cout << "time passed since last call = " << period << std::endl;
    ////////////////////////////////////////////////////////////////////////

    // Indices to pick out rows in robot Jacobian
    const int gaze_fixation_indices[] = {3, 4};
    const int all_other_dof_indices[] = {0, 1, 2, 5};
    const int all_other_dof_except_roll_indices[] = {0, 1, 2};

    ////////////////////////////////////////////////////////////////////////
    // Task priority 2 -- Fixation
    const Eigen::Matrix<double, 2, 7> gaze_fixation_jacobian = robot_jacobian(
        gaze_fixation_indices, Eigen::all);
    const Eigen::Vector<double, 2> gaze_fixation_commands = 
        curr_command_camera_frame_eigen(gaze_fixation_indices);

    Eigen::Vector<double, 7> joint_vels_fixation_task;
    Eigen::Matrix<double, 7, 7> nullspace_projector_fixation_task;
    Eigen::Matrix<double, 2, 7> gaze_fixation_augmented_jacobian;
    Eigen::Matrix<double, 7, 2> gaze_fixation_augmented_jacobian_pseudoinverse;

    computeSubtaskJointVels<1, 2>(
        gaze_fixation_jacobian,
        gaze_fixation_commands,
        joint_limit_avoidance_augmented_jacobian,
        joint_limit_avoidance_augmented_jacobian_pseudoinverse,
        joint_vels_joint_limit_avoidance,
        nullspace_projector_joint_limit_avoidance,
        joint_vels_fixation_task,
        nullspace_projector_fixation_task,
        gaze_fixation_augmented_jacobian,
        gaze_fixation_augmented_jacobian_pseudoinverse
    );
    
    // DEBUG
    // const ros::Time clock_after_fixation_task = ros::Time::now();
    // const double time_taken_after_fixation_task = (clock_after_fixation_task - curr_time).toSec();
    // std::cout << "time_taken_after_fixation_task = " << time_taken_after_fixation_task << std::endl;
    ////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////
    // Task priority 3 -- All other DOFs (translations + roll)
    const Eigen::Matrix<double, 4, 7> all_other_dof_jacobian = robot_jacobian(
        all_other_dof_indices, Eigen::all);
    const Eigen::Vector<double, 4> all_other_dof_commands = 
        curr_command_camera_frame_eigen(all_other_dof_indices);

    Eigen::Vector<double, 7> joint_vels_all_other_dof;
    Eigen::Matrix<double, 7, 7> nullspace_projector_all_other_dof;
    Eigen::Matrix<double, 4, 7> all_other_dof_augmented_jacobian;
    Eigen::Matrix<double, 7, 4> all_other_dof_augmented_jacobian_pseudoinverse;

    computeSubtaskJointVels<2, 4>(
        all_other_dof_jacobian,
        all_other_dof_commands,
        gaze_fixation_augmented_jacobian,
        gaze_fixation_augmented_jacobian_pseudoinverse,
        joint_vels_fixation_task,
        nullspace_projector_fixation_task,
        joint_vels_all_other_dof,
        nullspace_projector_all_other_dof,
        all_other_dof_augmented_jacobian,
        all_other_dof_augmented_jacobian_pseudoinverse
    );

    // DEBUG
    // const ros::Time clock_after_all_other_dof_task = ros::Time::now();
    // const double time_taken_after_all_other_dof_task = (clock_after_all_other_dof_task - curr_time).toSec();
    // std::cout << "time_taken_after_all_other_dof_task = " << time_taken_after_all_other_dof_task << std::endl;
    ////////////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////////////
    // Manipulablity in this form SUCKS. When it is a task with high priority
    // it is adverserial to lower tasks, as manipulablity index as potential
    // function is not Lyapunov stable. When it is a lower task, it gets
    // into weird saddle points. Anyway maintaining this task here for posterity.
    //
    // // Task priority X -- Maintain manipulability
    // Eigen::Vector<double, 1> manipulability_command;
    // manipulability_command[0] = -MANIPULABILITY_TASK_GAIN_;
    
    // Eigen::Matrix<double, 1, 7> manipulability_task_jacobian;
    // manipulability_task_jacobian.setZero();

    // // IMPORTANT NOTE
    // // As Eigen doesn't support Tensor yet,
    // // have to use array<Matrix> instead
    // // Here indexing robot_hessian[i, j, k] will be
    // // robot_hessian[k](i, j)
    // std::array<Eigen::Matrix<double, 6, 7>, 7> robot_hessian;
    // computeManipulatorHessianFromManipulatorJacobian(robot_jacobian, robot_hessian);

    // ///////////////////////////////
    // // DEBUG -- Print Hessian
    // // std::cout << "robot_jacobian" << std::endl;
    // // std::cout << robot_jacobian << std::endl;

    // // std::cout << "robot_hessian" << std::endl;
    // // for (int ix = 0; ix < 7; ix++)
    // // {
    // //     std::cout << "df_dqi_dq" << ix 
    // //     << std::endl << robot_hessian[ix] << std::endl;
    // // }
    // // std::cout << std::endl;
    // ///////////////////////////////

    // // TODO make this param
    // const bool MANIPULATOR_TASK_ONLY_ALONG_TRANS_ = false;
    
    // // NOTE could probably push each instance to templated code
    // // or use dynamic arrays (but dynamic will slow things down)
    // if (MANIPULATOR_TASK_ONLY_ALONG_TRANS_)
    // {
    //     const Eigen::Matrix<double, 3, 7> robot_jacobian_subset = 
    //         robot_jacobian({0, 1, 2}, Eigen::all);
    //     const Eigen::Matrix<double, 3, 3> robot_jacobian_robot_jacobian_T =
    //         robot_jacobian_subset * robot_jacobian_subset.transpose();
    //     Eigen::Map<const Eigen::Vector<double, 3 * 3>> robot_jacobian_robot_jacobian_T_flattened(
    //         robot_jacobian_robot_jacobian_T.data(), 3 * 3);

    //     const double manipulability_measure = std::sqrt(
    //         robot_jacobian_robot_jacobian_T.determinant()
    //     );
    //     for (int ix = 0; ix < 7; ix++)
    //     {
    //         const Eigen::Matrix<double, 3, 3> jacobian_hessian_T = 
    //             robot_jacobian_subset * robot_hessian[ix]({0, 1, 2}, Eigen::all).transpose();
    //         Eigen::Map<const Eigen::Vector<double, 3 * 3>> jacobian_hessian_T_flattened(
    //             jacobian_hessian_T.data(), 3 * 3);

    //         manipulability_task_jacobian(0, ix) = 
    //             - manipulability_measure * 
    //             robot_jacobian_robot_jacobian_T_flattened.dot(jacobian_hessian_T_flattened);
    //     }
    // }
    // else
    // {
    //     const Eigen::Matrix<double, 6, 6> robot_jacobian_robot_jacobian_T =
    //         robot_jacobian * robot_jacobian.transpose();
    //     Eigen::Map<const Eigen::Vector<double, 6 * 6>> robot_jacobian_robot_jacobian_T_flattened(
    //         robot_jacobian_robot_jacobian_T.data(), 6 * 6);

    //     const double manipulability_measure = std::sqrt(
    //         robot_jacobian_robot_jacobian_T.determinant()
    //     );
    //     if (manipulability_measure < 0.5)
    //     {
    //     for (int ix = 0; ix < 7; ix++)
    //     {
    //         const Eigen::Matrix<double, 6, 6> jacobian_hessian_T = 
    //             robot_jacobian * robot_hessian[ix].transpose();
    //         Eigen::Map<const Eigen::Vector<double, 6 * 6>> jacobian_hessian_T_flattened(
    //             jacobian_hessian_T.data(), 6 * 6);

    //         manipulability_task_jacobian(0, ix) = 
    //             - manipulability_measure * 
    //             robot_jacobian_robot_jacobian_T_flattened.dot(jacobian_hessian_T_flattened);
    //     }
    //     }
    // }
    // std::cout << manipulability_task_jacobian << std::endl;

    // Eigen::Vector<double, 7> joint_vels_manipulability_task;
    // Eigen::Matrix<double, 7, 7> nullspace_projector_manipulability_task;
    // Eigen::Matrix<double, 1, 7> manipulability_task_augmented_jacobian;
    // Eigen::Matrix<double, 7, 1> manipulability_task_augmented_jacobian_pseudoinverse;

    // computeSubtaskJointVels<4, 1>(
    //     manipulability_task_jacobian,
    //     manipulability_command,
    //     all_other_dof_augmented_jacobian,
    //     all_other_dof_augmented_jacobian_pseudoinverse,
    //     joint_vels_all_other_dof,
    //     nullspace_projector_all_other_dof,
    //     joint_vels_manipulability_task,
    //     nullspace_projector_manipulability_task,
    //     manipulability_task_augmented_jacobian,
    //     manipulability_task_augmented_jacobian_pseudoinverse
    // );
    // ////////////////////////////////////////////////////////////////////////

    // ////////////////////////////////////////////////////////////////////////
    // Example with dynamic consistency
    // Eigen::Vector<double, 7> joint_vels_all_other_dof;
    // Eigen::Matrix<double, 7, 7> nullspace_projector_all_other_dof;
    // Eigen::Matrix<double, 6, 7> all_other_dof_augmented_jacobian;
    // Eigen::Matrix<double, 7, 6> all_other_dof_augmented_jacobian_pseudoinverse;
    // computeDynamicallyConsistentSubtaskJointVels<1, 6>(
    //     robot_jacobian,
    //     curr_command_camera_frame_eigen,
    //     joint_limit_avoidance_augmented_jacobian,
    //     joint_limit_avoidance_augmented_jacobian_pseudoinverse,
    //     joint_vels_joint_limit_avoidance,
    //     nullspace_projector_joint_limit_avoidance,
    //     joint_vels_all_other_dof,
    //     nullspace_projector_all_other_dof,
    //     all_other_dof_augmented_jacobian,
    //     all_other_dof_augmented_jacobian_pseudoinverse
    // );
    // ////////////////////////////////////////////////////////////////////////

    curr_command_joint_vels_eigen_ = 
        joint_vels_joint_limit_avoidance + joint_vels_fixation_task + joint_vels_all_other_dof;
    last_time_nonrt_thread_done_ = ros::Time::now();
}

void TPJointVelocityController::computeManipulatorHessianFromManipulatorJacobian(
      const Eigen::Matrix<double, 6, 7>& robot_jacobian,
      std::array<Eigen::Matrix<double, 6, 7>, 7>& robot_hessian_r)
{
    // First just initialize the Hessian
    for (int jx = 0; jx < 7; jx++)
    {
        robot_hessian_r[jx].setZero();
    }

    const int translation_indices[] = {0, 1, 2};
    const int rotation_indices[] = {3, 4, 5};
    for (int jx = 0; jx < 7; jx++)
    {
        for (int ix = jx; ix < 7; ix++)
        {
            // const Eigen::Vector<double, 3> J_tra_jx = 
            //     robot_jacobian(translation_indices, jx);
            const Eigen::Vector<double, 3> J_rot_jx = 
                robot_jacobian(rotation_indices, jx);
            const Eigen::Vector<double, 3> J_tra_ix = 
                robot_jacobian(translation_indices, ix);
            const Eigen::Vector<double, 3> J_rot_ix = 
                robot_jacobian(rotation_indices, ix);

            robot_hessian_r[jx](translation_indices, ix) = J_rot_jx.cross(J_tra_ix);
            robot_hessian_r[jx](rotation_indices, ix) = J_rot_jx.cross(J_rot_ix);

            if (ix != jx)
            {
                robot_hessian_r[ix](translation_indices, jx) = 
                    robot_hessian_r[jx](translation_indices, ix);
            }
        }
    }
}

template<int TASK_DIM>
void TPJointVelocityController::computeMaintaskJointVels(
    const Eigen::Matrix<double, TASK_DIM, 7>& main_task_jacobian,
    const Eigen::Vector<double, TASK_DIM>& main_task_commands,
    Eigen::Vector<double, 7>& main_task_computed_joint_vels_r,
    Eigen::Matrix<double, 7, TASK_DIM>& main_task_jacobian_pseudoinverse_r
)
{
    moorePenrosePseudoInverse<TASK_DIM>(main_task_jacobian, main_task_jacobian_pseudoinverse_r);
    main_task_computed_joint_vels_r = main_task_jacobian_pseudoinverse_r * main_task_commands;
}

template<int PREV_TASK_DIM, int TASK_DIM>
void TPJointVelocityController::computeSubtaskJointVels(
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
)
{
    this_subtask_nullspace_projector_r = prev_subtask_nullspace_projector *
                                         (Eigen::Matrix<double, 7, 7>::Identity() -
                                         prev_subtask_jacobian_pseudoinverse * prev_subtask_jacobian);
    this_subtask_augmented_jacobian_r = this_subtask_jacobian * this_subtask_nullspace_projector_r;

    moorePenrosePseudoInverse<TASK_DIM>(
        this_subtask_augmented_jacobian_r, this_subtask_augmented_jacobian_pseudoinverse_r);

    // Return computed joint velocities
    this_subtask_computed_joint_vels_r = this_subtask_nullspace_projector_r *
                                         this_subtask_augmented_jacobian_pseudoinverse_r *
                                         (this_subtask_commands - 
                                         this_subtask_jacobian * prev_subtask_joint_vels);
}

template<int PREV_TASK_DIM, int TASK_DIM>
void TPJointVelocityController::computeFullRankSubtaskJointVels(
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
)
{
    this_subtask_nullspace_projector_r = prev_subtask_nullspace_projector *
                                         (Eigen::Matrix<double, 7, 7>::Identity() -
                                         prev_subtask_jacobian_pseudoinverse * prev_subtask_jacobian);
    this_subtask_augmented_jacobian_r = this_subtask_jacobian * this_subtask_nullspace_projector_r;

    vanillaJacobianPseudoInverse<TASK_DIM>(
        this_subtask_augmented_jacobian_r, this_subtask_augmented_jacobian_pseudoinverse_r);

    // Return computed joint velocities
    this_subtask_computed_joint_vels_r = this_subtask_nullspace_projector_r *
                                         this_subtask_augmented_jacobian_pseudoinverse_r *
                                         (this_subtask_commands - 
                                         this_subtask_jacobian * prev_subtask_joint_vels);
}

template<int PREV_TASK_DIM, int TASK_DIM>
void TPJointVelocityController::computeDynamicallyConsistentSubtaskJointVels(
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
)
{
    this_subtask_nullspace_projector_r = prev_subtask_nullspace_projector *
                                         (Eigen::Matrix<double, 7, 7>::Identity() -
                                         prev_subtask_jacobian_pseudoinverse * prev_subtask_jacobian);
    this_subtask_augmented_jacobian_r = this_subtask_jacobian * this_subtask_nullspace_projector_r;

    // vanillaJacobianPseudoInverse<TASK_DIM>(
    //     this_subtask_augmented_jacobian_r, this_subtask_augmented_jacobian_pseudoinverse_r);
    dynamicallyConsistentJacobianPseudoInverse<TASK_DIM>(
        this_subtask_augmented_jacobian_r, this_subtask_augmented_jacobian_pseudoinverse_r);

    // Return computed joint velocities
    this_subtask_computed_joint_vels_r = this_subtask_nullspace_projector_r *
                                         this_subtask_augmented_jacobian_pseudoinverse_r *
                                         (this_subtask_commands - 
                                         this_subtask_jacobian * prev_subtask_joint_vels);
}

template<int TASK_DIM>
void TPJointVelocityController::vanillaJacobianPseudoInverse(
    const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
    Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
)
{
    Eigen::Matrix<double, TASK_DIM, TASK_DIM> jacobian_jacobian_T = 
        jacobian * jacobian.transpose();
    
    // Return pseudoinverse
    jacobian_pseudoinverse_r = 
        jacobian.transpose() * jacobian_jacobian_T.inverse();
}

template<int TASK_DIM>
void TPJointVelocityController::moorePenrosePseudoInverse(
    const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
    Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
)
{
    Eigen::Matrix<double, TASK_DIM, TASK_DIM> jacobian_jacobian_T = 
        jacobian * jacobian.transpose();
    
    // Return pseudoinverse
    jacobian_pseudoinverse_r = 
        jacobian.transpose() * jacobian_jacobian_T.inverse();

    // NOTE JacobiSVD DOES NOT allow static matrix because some
    // weird way size of U and V matrices are computed. So converting to dynamic
    // More details: https://gitlab.com/libeigen/eigen/-/issues/2051
    Eigen::JacobiSVD<Eigen::MatrixXd> 
        svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double tolerance = std::numeric_limits<double>::epsilon()
        * 7         // std::max(jacobian.cols(), jacobian.rows())
        * svd.singularValues().array().abs()(0);

    // Pseudoinverse = V * (1 / non-zero-singular-values) * U^T
    jacobian_pseudoinverse_r = 
        svd.matrixV() 
        * (
            svd.singularValues().array().abs() > tolerance
          ).select(
              svd.singularValues().array().inverse(), 0
              ).matrix().asDiagonal() 
        * svd.matrixU().adjoint();
}

template<int TASK_DIM>
void TPJointVelocityController::dynamicallyConsistentJacobianPseudoInverse(
    const Eigen::Matrix<double, TASK_DIM, 7>& jacobian,
    Eigen::Matrix<double, 7, TASK_DIM>& jacobian_pseudoinverse_r
)
{
    const std::array<double, 49> inertia_matrix_array = model_handle_->getMass();

    Eigen::Map<const Eigen::Matrix<double, 7, 7>> inertia_matrix(inertia_matrix_array.data());
    const Eigen::Matrix<double, 7, 7> inertia_matrix_inverse = inertia_matrix.inverse();

    Eigen::Matrix<double, TASK_DIM, TASK_DIM> jacobian_jacobian_T = 
        jacobian * inertia_matrix_inverse * jacobian.transpose();
    
    // Return pseudoinverse
    jacobian_pseudoinverse_r = 
        inertia_matrix_inverse * jacobian.transpose() * jacobian_jacobian_T.inverse();
}


void TPJointVelocityController::gazeLockCommandCallback(
    const geometry_msgs::TwistStampedConstPtr& msg)
{
    {
        std::lock_guard<std::mutex> _(gaze_lock_commands_mutex_);
        requested_wX_ = msg->twist.angular.x;
        requested_wY_ = msg->twist.angular.y;
        last_rcvd_time_gaze_lock_commands_ = msg->header.stamp;
    }
}

void TPJointVelocityController::heuristicCommandCallback(
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

void TPJointVelocityController::approachCommandCallback(
    const geometry_msgs::TwistStampedConstPtr& msg)
{
    {
        std::lock_guard<std::mutex> _(approach_commands_mutex_);
        requested_vZ_ = msg->twist.linear.z;
        last_rcvd_time_approach_commands_ = msg->header.stamp;
    }
}

void TPJointVelocityController::smoothTrajectory(const Eigen::Vector<double, 7>& desired_values,
                                                 Eigen::Vector<double, 7>& smoothed_values_r)
{
    franka::RobotState robot_state = state_handle_->getRobotState();
    auto prev_desired_values = robot_state.dq_d;

    const double robot_control_freq = 1e3; // Hz

    // Choose constant max acceleration now, although Panda allows
    // different accelerations for different joints
    const double max_acc_per_joint = 5 / 
            robot_control_freq / robot_control_freq; // rad/s^2 --> rad/tick^2
    for (size_t i = 0; i < 7; i++)
    {        
        double desired_difference = (desired_values[i] - prev_desired_values[i]) / robot_control_freq;
        smoothed_values_r[i] =
            prev_desired_values[i] +
            // std::max(std::min(desired_difference, max_acc_per_joint[i]), -max_acc_per_joint[i]) * 1e-3;
            std::max(
                std::min(
                    desired_difference, max_acc_per_joint), -max_acc_per_joint) * robot_control_freq;
    }
}


}  // namespace events_shaping_controller

PLUGINLIB_EXPORT_CLASS(events_shaping_controller_control::TPJointVelocityController,
                       controller_interface::ControllerBase)