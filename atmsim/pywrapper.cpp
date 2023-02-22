
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "atm_sim.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(MyLib, m) {
    m.doc() = "optional module docstring";

    py::class_<ATMSim>(m, "ATMSim")
    .def(py::init<std::string, std::string, bool, int, float>())  
    .def("step", &ATMSim::step, py::call_guard<py::gil_scoped_release>())

    // .def_readonly("v_data", &ATMSim::v_data, byref)
    // .def_readonly("v_gamma", &ATMSim::v_gamma, byref)
    ;
}