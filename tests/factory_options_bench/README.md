# factory_options benchmark

Archived test for comparing static inline member variable vs static inline member function for Tensor factory_options.

## Files

- blob_static_var.cpp — test source using factory_options (static member variable)
- blob_static_func.cpp — test source using parameter_options() (static member function)
- core_static_var.mpp — copy of original src/typetorch/tensor/core.mpp
- core_static_func.mpp — patched variant with parameter_options()
- factory_static_func.mpp — patched variant of factory.mpp
- run_factory_compare.sh — runner script

## How to run

    cd /work/7/uw07387/libtorch-reflect-wrapper
    source scripts/env.sh
    tests/factory_options_bench/run_factory_compare.sh

See docs/factory-options-static-var-vs-func.md for results.
