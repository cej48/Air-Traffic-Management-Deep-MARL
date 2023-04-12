cmake_minimum_required(VERSION 3.10)

project(PyATMSim)
set(CMAKE_CXX_STANDARD 20)
set(PYBIND11_PYTHON_VERSION 3.6)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(${PROJECT_NAME} ./atmsim/pywrapper.cpp
./atmsim/traffic.cpp
./atmsim/atm_sim.cpp
./atmsim/atmosphere.cpp
./atmsim/atm_interface.cpp
./atmsim/utils.cpp)

find_package(SFML 2.5.1
COMPONENTS 
system window graphics network audio REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics)

file(COPY ./atmsim/airport_data.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ./atmsim/environment_boundaries.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ./atmsim/arial.ttf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(nlohmann_json 3.11.2 REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PUBLIC tbb)