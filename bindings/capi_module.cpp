import std;
import python;
import tenspec.capi_bridge;

namespace py = python;

namespace
{

struct python_function
{
	::std::string name{};
	::std::string doc{};
	py::PyCFunction invoke{};
	py::PyMethodDef method{};
};

::std::deque<python_function> functions;

void collect_function(tenspec_python_function const *function, void *) noexcept
{
	if (!function || !function->name || !function->invoke)
	{
		return;
	}

	auto &record{functions.emplace_back()};
	record.name = function->name;
	record.doc = function->doc ? function->doc : "";
	record.invoke = function->invoke;
}

py::PyModuleDef module = py::module_def(
	"tenspec_capi",
	"Minimal typed tensor bindings backed by libtorch and the Python C API.");

} // namespace

extern "C" __attribute__((visibility("default"))) auto PyInit_tenspec_capi()
	-> py::PyObject *
{
	py::PyObject *torch = py::PyImport_ImportModule("torch");
	if (!torch)
	{
		return nullptr;
	}
	py::decref(torch);

	auto *module_object{py::module_create(&module)};
	if (!module_object)
	{
		return nullptr;
	}

	functions.clear();
	tenspec_python_for_each_function(collect_function, nullptr);

	for (auto &function : functions)
	{
		function.method = py::PyMethodDef{
			.ml_name = function.name.c_str(),
			.ml_meth = function.invoke,
			.ml_flags = py::meth_varargs,
			.ml_doc = function.doc.empty() ? nullptr : function.doc.c_str(),
		};

		auto *callable{py::PyCFunction_NewEx(&function.method, nullptr, nullptr)};
		if (!callable)
		{
			py::decref(module_object);
			return nullptr;
		}

		if (py::PyModule_AddObject(module_object, function.name.c_str(), callable) < 0)
		{
			py::decref(callable);
			py::decref(module_object);
			return nullptr;
		}
	}

	return module_object;
}
