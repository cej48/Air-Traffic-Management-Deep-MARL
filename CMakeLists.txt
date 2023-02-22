cmake_minimum_required(VERSION 3.0.0)
project(atm_sim VERSION 0.1.0)

include(CTest)
enable_testing()

add_compile_options(-O3)
add_executable(atm_sim 
                ./atmsim/main.cpp
                ./atmsim/traffic.cpp
                ./atmsim/atm_sim.cpp
                ./atmsim/atmosphere.cpp
                ./atmsim/atm_interface.cpp
                ./atmsim/utils.cpp
)

file(COPY ./atmsim/airport_data.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ./atmsim/environment_boundaries.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ./atmsim/arial.ttf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(SFML 2.5.1
COMPONENTS 
system window graphics network audio REQUIRED)
target_link_libraries(atm_sim sfml-graphics)

find_package(nlohmann_json 3.11.2 REQUIRED)
target_link_libraries(atm_sim nlohmann_json::nlohmann_json)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (atm_sim Eigen3::Eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
