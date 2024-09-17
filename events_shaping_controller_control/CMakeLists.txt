cmake_minimum_required(VERSION 3.4)
project(events_shaping_controller_control)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    events_shaping_controller_msgs
    dvs_msgs
    hardware_interface
    tf
    tf_conversions
    pluginlib
    realtime_tools
    rospy
)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.7.0 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Generate dynamic reconfigure stuff
# generate_dynamic_reconfigure_options(
#   cfg/TODO.cfg
# )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES events_shaping_controller_control
  CATKIN_DEPENDS
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    events_shaping_controller_msgs
    hardware_interface
    tf
    tf_conversions
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)

add_library(events_shaping_controller_control
        src/events_shaping_controller_cartesian_velocity_controller.cpp
        src/events_shaping_controller_task_priority_joint_velocity_controller.cpp
)


# add_dependencies(events_shaping_controller_control
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
#   ${PROJECT_NAME}_generate_messages_cpp
#   ${PROJECT_NAME}_gencpp
# #   ${PROJECT_NAME}_gencfg
# )

target_link_libraries(events_shaping_controller_control PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
)

include(ExternalProject)

# message(WARNING "Installation directory: ${CMAKE_BINARY_DIR}/image-warped-events}")

ExternalProject_Add(
    image-warped-events
    PREFIX ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY https://github.com/juanangd/image-warped-events # Replace with the actual repository URL
    # GIT_TAG main
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND "" # No configure step needed for Python packages
    BUILD_COMMAND ""     # No build step needed for Python packages
    INSTALL_COMMAND ${Python3_EXECUTABLE} -m pip install ${CMAKE_BINARY_DIR}/src/image-warped-events
)

target_include_directories(events_shaping_controller_control SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(events_shaping_controller_control PUBLIC
  include
)
include_directories(events_shaping_controller_control PUBLIC
  include
)

# Installation
install(TARGETS events_shaping_controller_control
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES events_shaping_controller_controllers_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Tools (copied from franka_example_controllers)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(events_shaping_controller_control FILES ${SOURCES} ${HEADERS})
  add_tidy_target(events_shaping_controller_control
    FILES ${SOURCES}
    DEPENDS events_shaping_controller_control
  )
endif()
