#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/torch.h>
#include <torch/csrc/autograd/python_variable.h>

extern "C" bool typetorch_thp_tensor_check(void *object) noexcept
{
	return ::THPVariable_Check(static_cast<PyObject *>(object)) != 0;
}

extern "C" void typetorch_thp_tensor_unpack(void *object, void *out_tensor)
{
	auto *out = static_cast<torch::Tensor *>(out_tensor);
	*out = ::THPVariable_Unpack(static_cast<PyObject *>(object));
}

extern "C" void *typetorch_thp_tensor_wrap(void *tensor)
{
	auto *value = static_cast<torch::Tensor *>(tensor);
	return static_cast<void *>(::THPVariable_Wrap(std::move(*value)));
}