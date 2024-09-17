


# Shaping the Spatio-Temporal Structure of Event-based camera stream for robot control. 

Velocity based control of a Franka Panda robot through Event-based camera spatio-temporal streams active Shaping. 
![EventShaper](event_shaper_img.png | width=300)

Features:
- Runs on ROS (Python and C++ nodes)

## Dependencies
- ROS Noetic / Melodic
- libfranka 0.7.0+, franka_ros (Modified to publish Jacobian: https://github.com/aravindbattaje/franka_ros.git)
- PyTorch 1.9 (system installation; doesn't play well with conda and RoboStack)
- rpg_ros_driver (Modified to publish events as flattened arrays: https://github.com/juanangd/rpg_dvs_ros)
- image-warped-events (Python library implemented to compute the sharpness of IWEs and their corresponding derivatives; it will get automatically installed when cmake build https://github.com/juanangd/image-warped-events)

## Hardware

- Franka Emika Panda robot
- Control PC with realtime kernel
- DAVIS 346: DVS event camera with included active pixel frame sensor
- USB 3.0 extension cable

## Installation

1. Install libfranka and franka ros (https://frankaemika.github.io/docs/installation_linux.html)
2. Crete a catkin_ws (http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
3. Install rpg_ros_driver  (https://github.com/juanangd/rpg_dvs_ros)
4. In the `catkin_ws/src`, clone this repo git (`git clone --recursive https://github.com/juanangd/events_shaping_controller.git`)
5. From `catkin_ws`, run `catkin build`

## How to run?

### Fixation constrained motion through A-CMAX + Centroids visual servoing

The following roslaunch command executes the needed nodes to achieve fixation by combining vx and wy (by default) through purely A-CMAX, centroid-based visual servoing or both : 

`roslaunch events_shaping_controller_control all_omega_optm.launch config_file:=$(find events_shaping_controller_control)/cfg/your_custom_config.yaml`

Available config files:

 -  `all_config_optm_wy_acmax_constant.yaml`: Using purely a-cmax to achieve fixation when a smoothed square signal is used by the heuristic commander (to set vx)   
 - `all_config_optm_wy_acmax_sine.yaml`: Using purely a-cmax to achieve fixation when a sine signal is used by the heuristic commander (to set vx)
 - `all_config_optm_wy_both_constant.yaml`: Using a-cmax + centroid to achieve fixation when a smoothed square signal is used by the heuristic commander (to set vx)
 -   `all_config_optm_wy_both_both_sine.yaml`: Using a-cmax + centroid to achieve fixation when a sine signal is used by the heuristic commander (to set vx)
    - `all_config_optm_wy_centroid_constant.yaml`: Using centroid-based visual servoing to achieve fixation when a smoothed square signal is used by the heuristic commander (to set vx)    
    - `all_config_optm_wy_centroid_sine.yaml`: Using centroid-based visual servoing to achieve fixation when a sine signal is used by the heuristic commander (to set vx)    

##### Parameters of the gaze lock commander:

| Argument               | Meaning                                                                                                | Default Value |
|------------------------|--------------------------------------------------------------------------------------------------------|-------|
| `analyze_only_bounding_box`             | It only analyze the events from the ROI (normally with their pa meters also configurable in the config) | `True`  |
| `num_events_threshold` | Minimum number of events needed to compute compensatory command (If not reached, the compensatory command will be set to the previous value) | `300` 
| `maximum_num_events_to_process` | Maximum number of events that will be processed by the node. If the current number is larger, the data is downsampled | `600`
| `alpha_ema`  | Value of the alpha factor of the Exponential smoothing (https://en.wikipedia.org/wiki/Exponential_smoothing). `smoothed_jacobian = alpha_ema * current_jacobian + (1 - alpha_ema) * old_jacobian` It was never really used| `1.0`
| `learning_rate` | A-CMAX learning rate (referred in the thesis as $\gamma$) Its value will depend on the sharpness function type| `0.006` (for `image_area`)
| `sharpness_function_type` | Sharpness function to be applied to the IWE. Available Functions: `magnitude_gradient`, `magnitude_hessian`, `possion`, `variance`, `mean_absolute_deviation`, `number_pixels_activated`, `entropy`, `image_range`, `moran_index`, `geary_contiguity_ratio`| `variance`
| `params_to_evaluate` | Over which of the 3 DoF the derivative should be evaluated. When Vx + wy is used the derivative is df/dwy | `[False, True, False ]` 
| `kp_control_jacobian` | weight of the jacobian control law | It depends
| `kp_control_centroid` | weight of the centroid control law |  It depends
| `motion_model` | Specify if the motion model is  `rotation` or `translation` | `rotation` 
| `publish_jacobian` | Determines if the jacobian values should be published | `True` 
| `publish_events_analyzed` | It determines if the events analyzed at every optimization steps need to be sent as `dvs_msgs/msgs/EventArrayFlattened` | `False`
| `jacobian_clipping_value` | Clipping value of the jacobian computation | `20`

### Constant Visual Divergence constrained motion

The following roslaunch command executes the needed nodes to achieve the visual constant divergence constrained motion with exponentially decaying velocity v_z  as the camera approaches the surface:

`roslaunch events_shaping_controller_control all_omega_optm.launch config_file:=$(find events_shaping_controller_control)/cfg/your_custom_config.yaml`

Available config files:

 - `all_config_approach_exp1.yaml`: It runs the experiment 1 (as referred on the thesis) where D* = -0.15
 - `all_config_approach_exp1.yaml`:  It runs the experiment 1 (as referred on the thesis) where D* = -0.15

##### Parameters of the approach commander:

| Argument               | Meaning                                                                                                | Default Value |
|------------------------|--------------------------------------------------------------------------------------------------------|-------|
| `publish_jacobian` | Determines if the jacobian values should be published | `True` 
| `publish_events_analyzed` | It determines if the events analyzed at every optimization steps need to be sent as `dvs_msgs/msgs/EventArrayFlattened` | `False`
| `analyze_only_bounding_box`             | It only analyze the events from the ROI (normally with their parameters also configurable in the config) | `True`  |
| `num_events_threshold` | Minimum number of events needed to compute compensatory command (If not reached, the compensatory command will be set to the previous value) | `100` 
| `sliding_window_time_jacobian` | Length (s) of the time window to be analyze | `0.1`
| `maximum_num_events_to_process` | Maximum number of events that will be processed by the node. If the current number is larger, the data is downsampled | `100000`
| `alpha_ema`  | Value of the alpha factor of the Exponential smoothing (https://en.wikipedia.org/wiki/Exponential_smoothing). `smoothed_jacobian = alpha_ema * current_jacobian + (1 - alpha_ema) * old_jacobian` It was never really used| `0.7`
|`jacobian_computation_freq` | Frequency (HZ) of the computation loop: It buffers the sliding window and computes the jacobian | `50`
| `learning_rate` | A-CMAX learning rate (referred in the thesis as $\gamma$) Its value will depend on the sharpness function type| `7` (for `variance`)
| `max_commander_vel_saturation` | Maximum value of Vz that can be set (in m/s) | `0.3`
| `sharpness_function_type` | Sharpness function to be applied to the IWE. Available Functions: `magnitude_gradient`, `magnitude_hessian`, `possion`, `variance`, `mean_absolute_deviation`, `number_pixels_activated`, `entropy`, `image_range`, `moran_index`, `geary_contiguity_ratio`| `variance`
| `velocity_gain`| Initial gain velocity. It will be modulated through the optimization steps | `0.15`
| `divergence_rate_target` | Divergence rate target we are aiming at |  ` -0.15`
| `init_smoothing_time` | Time (in seconds) that needs the robot to reach a velocity from 0 ms/s to the `velocity_gain` using a spline | 0.5
| `initial_measure_distance` | Externally Measured distance at the beginning of the experiment  | It depends

### Heuristic mode

The heuristic node consist of commands that control the linear translation and other movements that were used in the experimentation stage. It was mainly used to achieve the fixation command as lateral motion had to be induced to enable compensatory commands. 

##### Parameters of the heuristic commander:

| Argument               | Meaning                                                                                                | Default Value |
|------------------------|--------------------------------------------------------------------------------------------------------|-------|
| `ground_truth_distance` | (Debug) It was used to plot the actual ground truth compensatory command | It depends 
| `gain_vxy_modulation_based_centroids` | Used in the exploratory phase to modulate the compensatory motion with centroid-based information | `0.`
| `current_heuristic_mode` | Combination of heuristic modes used| It depends 


To specify what combination of heuristic motions has to be used, set the parameter `current_heuristic_mode` as follows: `{modetype1_mode1}+{modetype2_mode2}` . For example `cycle_sinX+perturbation_sinZ` means that two modes will used: `SinX` from the modetype `cycle` and `sinZ` from the nodetype `perturbation`.  
The modetypes are the following:
cycle

| Modetype | Submode | What does it do? |
|------|---------|------------------|
| `cycle` | `sinX`  | Produces lateral sinusoidal motion with a frequency of `freq_heuristic_cycle_control_commands` Hz in the horizontal direction with a gain of `gain_heuristic_cycle_control_commands`  m/s |
|  | `sinY` | Produces lateral sinusoidal motion with a frequency of `freq_heuristic_cycle_control_commands` Hz in the vertical direction with a gain of `gain_heuristic_cycle_control_commands`  m/s  |
|  | `circularCW`  | Produces a circular linear motion with clockwise direction with a frequency of `freq_heuristic_cycle_control_commands` Hz and a gain of `gain_heuristic_cycle_control_commands`  m/s |
| | `circularCCW`  | Produces a circular linear motion with counter-clockwise direction with a frequency of `freq_heuristic_cycle_control_commands` Hz and a gain of `gain_heuristic_cycle_control_commands`  m/s |
| | `smoothConstantX` | Produces horizontal lateral motion with a smoothed square wave profile with a frequency of `freq_heuristic_cycle_control_commands` Hz and a gain of `gain_heuristic_cycle_control_commands`  m/s  |
| | `smoothConstantY` | Produces vertical lateral motion with a smoothed square wave profile with a frequency of `freq_heuristic_cycle_control_commands` Hz and a gain of `gain_heuristic_cycle_control_commands`  m/s  |
 |`perturbation` | `sinZ` | Add some motion in the Z axis. It was used to force the events generation. It has a Siusoidal profile with  a gain of  `gain_heuristic_perturbation_control_commands` m/s and a frequency of `freq_heuristic_perturbation_control_commands`
 | | `constant` | It adds a constant velocity in the Z axis with a gain of `gain_heuristic_approach_control_commands` m/s
 | | `sinZ` | It adds a cyclic velocity in the Z axis with a `gain_heuristic_approach_control_commands` m/s and a frequency of `freq_heuristic_approach_control_commands` that every `approach_command_cycle_interval` times, get applied  an extra gain of `approach_extra_gain` for the positive part of the signal.  

### Other Launch files 

 - `events_shaping_controller_control/cfg/move_to_start.launch`: It brings the robot to the home pose configuration
 - `events_shaping_controller_control/cfg/all_optimization_distance.launch` It achieves the fixational motion by maximimizing the sharpness of image of accumulated events over the distance using the fixation constrain.
 - `events_shaping_controller_control/cfg/publish_image_with_bounding_box`: Publish the frame-based camera with the bounding box used for analyzing the events 

## Modified franka_ros
(It is a legacy feature from gcvs (https://git.tu-berlin.de/scioi/project-002/gcvs))
As state estimation requires robot Jacobians, here is the latest release of franka_ros (0.9.0) modified to publish Jacobians. If you don't use this version, distance estimation and hence approach controller does not work!

Modified franka_ros: [https://github.com/aravindbattaje/franka_ros.git](https://github.com/aravindbattaje/franka_ros.git)


##  Master Thesis Reference 

```
@masterthesis{gomez_shaping24,
  title        = {Shaping Spatio-Temporal Event-Based Camera Stream for Robot Control},
  author       = {Juan Antonio Gomez Daza},
  year         = 2024,
  month        = {August},
  address      = {Berlin, Germany},
  note         = {Available at \url{https://tubcloud.tu-berlin.de/s/eZ56onpPwx5JwMR}},
  school       = {Technical University Berlin},
  type         = {Master's thesis}
}
```
