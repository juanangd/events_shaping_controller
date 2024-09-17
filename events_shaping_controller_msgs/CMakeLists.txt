cmake_minimum_required(VERSION 3.4)
project(events_shaping_controller_msgs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    message_generation
    sensor_msgs
)

add_message_files(FILES
  ImgAccumulatedEventsDistance.msg
  Vector2.msg
  OpticalFlowEstimation.msg
  Vector2Stamped.msg
  StampedEventPacket.msg
)

# Add action files here when needed

generate_messages(
  DEPENDENCIES 
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    std_msgs
    message_runtime
)


