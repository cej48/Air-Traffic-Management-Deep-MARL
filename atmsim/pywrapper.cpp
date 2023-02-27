
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "atm_sim.h"
#include "traffic.h"
#include "airport.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(PyATMSim, m) {
    m.doc() = "optional module docstring";

    py::class_<ATMSim>(m, "ATMSim")
    .def(py::init<std::string, std::string, bool, int, float>())
    .def(py::init<ATMSim*, bool>())
    .def("step", &ATMSim::step, py::call_guard<py::gil_scoped_release>())
    .def("copy_from_other", &ATMSim::copy_from_other)
    .def_readonly("traffic", &ATMSim::traffic, byref)
    
    .def_readonly("observation_space", &ATMSim::observation_space, byref)
    .def_readonly("action_space", &ATMSim::action_space, byref)
    ;
    py::class_<Traffic> (m, "Traffic")
    .def_readonly("position", &Traffic::position, byref)
    .def_readonly("heading", &Traffic::heading, byref)
    .def_readonly("speed", &Traffic::speed, byref)
    .def_readonly("callsign", &Traffic::callsign, byref)
    .def_readonly("destination_hdg", &Traffic::destination_hdg, byref)
    .def_readonly("destination", &Traffic::destination, byref)
    ;

    py::class_<Airport> (m, "Airport")
    .def_readonly("position", &Airport::position, byref)
    ;

}
