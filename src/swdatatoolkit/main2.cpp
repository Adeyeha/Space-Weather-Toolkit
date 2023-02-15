#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int times(int i, int j) {
    return i * j;
}

namespace py = pybind11;

PYBIND11_MODULE(_core2, m) {
    m.doc() = R"pbdoc(
        Pybind11 example2 plugin
        ------------------------
        .. currentmodule:: swdatatoolkit._core2
        .. autosummary::
           :toctree: _generate


    )pbdoc";

    m.def("times", &times, R"pbdoc(
        Multiply two numbers
        Some other explanation about the times function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = "Main2";//MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}