cmake_minimum_required(VERSION 3.4)
project(events_shaping_controller_franka_description)

find_package(catkin REQUIRED)
catkin_package(CATKIN_DEPENDS xacro)

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY robots
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
