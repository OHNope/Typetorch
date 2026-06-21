set_project("typetorch")
set_version("0.1.0")
set_languages("c++26")
set_policy("build.c++.modules.gcc.cxx11abi", true)

local local_config = {}

function typetorch_config(config)
    local_config = config or {}
end

if os.isfile("typetorch.local.lua") then
    includes("typetorch.local.lua")
end

local function config_value(name, env_name)
    local env_value = os.getenv(env_name or name)
    if env_value and env_value ~= "" then
        return env_value
    end
    local value = local_config[name]
    if value ~= nil and value ~= "" then
        return value
    end
    return nil
end

local function detect_gcc_relocate_ldflags()
    local configured_flags = config_value("gcc_relocate_ldflags", "TYPETORCH_GCC_RELOCATE_LDFLAGS")
    if configured_flags then
        return configured_flags
    end

    local candidates = {}
    local gcc_root = config_value("gcc_root", "GCC_ROOT")
    local configured_toolchain = config_value("gcc_toolchain_root", "GCC_TOOLCHAIN_ROOT")

    if configured_toolchain then
        table.insert(candidates, configured_toolchain)
    end

    if gcc_root and gcc_root ~= "" then
        local triplet = config_value("gcc_triplet", "GCC_TRIPLET") or "x86_64-focal-linux-gnu"
        table.insert(candidates, path.join(path.directory(gcc_root), "x-tools", triplet))
    end

    for _, root in ipairs(candidates) do
        local triplet = config_value("gcc_triplet", "GCC_TRIPLET") or "x86_64-focal-linux-gnu"
        local version = config_value("gcc_version", "GCC_VERSION") or "16.1.0"
        local frontend_dir = path.join(root, "libexec", "gcc", triplet, version)
        local gcc_lib_dir = path.join(root, "lib", "gcc", triplet, version)
        local sysroot = config_value("gcc_sysroot", "GCC_SYSROOT") or path.join(root, triplet, "sysroot")

        if os.isfile(path.join(frontend_dir, "cc1plus")) and
           os.isfile(path.join(gcc_lib_dir, "crtbeginS.o")) and
           os.isfile(path.join(sysroot, "usr", "lib", "crti.o")) then
            return format(
                "-B%s/ -B%s/ -B%s/ -B%s/ --sysroot=%s",
                frontend_dir,
                gcc_lib_dir,
                path.join(sysroot, "usr", "lib"),
                path.join(sysroot, "lib"),
                sysroot
            )
        end
    end

    return nil
end

local gcc_relocate_ldflags = detect_gcc_relocate_ldflags()
if gcc_relocate_ldflags and gcc_relocate_ldflags ~= "" then
    for flag in gcc_relocate_ldflags:gmatch("%S+") do
        add_ldflags(flag, {force = true})
        add_shflags(flag, {force = true})
    end
end

add_rules("mode.debug", "mode.release")
add_repositories("local-repo .")

add_cxxflags(
    "-fmodules",
    "-freflection",
    "-Wall",
    "-Wextra",
    "-fvisibility=hidden",
    {force = true}
)

local configured_python_include = config_value("python_include_dir", "PYTHON_INCLUDE_DIR")
if configured_python_include then
    add_cxxflags("-isystem" .. configured_python_include, {force = true})
    add_sysincludedirs(configured_python_include)
end

add_requires("libtorch-bin 2.8.0+cu128", {alias = "libtorch"})

local typetorch_module_files = {
    "src/libtorch.mpp",
    "src/typetorch_types.mpp",
    "src/typetorch_torch_mappings.mpp",
    "src/typetorch_shape_meta.mpp",
    "src/typetorch_common.mpp",
    "src/typetorch_runtime_checks.mpp",
    "src/typetorch_tensor.mpp",
    "src/typetorch.mpp"
}

local typetorch_examples_module_files = {
    "src/typetorch_examples.mpp",
    "src/examples.cpp",
}

local fast_io_files = {
    "src/fast_io.mpp",
    "src/fast_io.cpp"
}

local common_debug_cxxflags = {
    "-O0",
    "-g3",
    "-fno-inline",
    "-fno-omit-frame-pointer"
}

local common_release_cxxflags = {
    "-O3",
    "-march=native",
    "-flto"
}

local common_release_ldflags = {
    "-flto"
}

local sanitizer_flags = {
    "-fsanitize=address,undefined"
}

local function add_file_list(files)
    for _, file in ipairs(files) do
        add_files(file)
    end
end

local function add_typetorch_modules()
    add_file_list(typetorch_module_files)
end

local function add_typetorch_examples_modules()
    add_file_list(typetorch_examples_module_files)
end

local function add_fast_io()
    add_file_list(fast_io_files)
end

local function add_cxxflag_list(flags)
    for _, flag in ipairs(flags) do
        add_cxxflags(flag)
    end
end

local function add_ldflag_list(flags)
    for _, flag in ipairs(flags) do
        add_ldflags(flag)
    end
end

local function add_common_mode_flags()
    if is_mode("debug") then
        add_cxxflag_list(common_debug_cxxflags)
        add_defines("TYPETORCH_DEBUG")

        if get_config("sanitizers") then
            add_cxxflag_list(sanitizer_flags)
            add_ldflag_list(sanitizer_flags)
        end
    elseif is_mode("release") then
        add_cxxflag_list(common_release_cxxflags)
        add_ldflag_list(common_release_ldflags)
        add_defines("TYPETORCH_RELEASE")
    end
end

local function default_python()
    local python_env = os.getenv("PYTHON")
    if python_env then
        return python_env
    end

    for _, candidate in ipairs({".venv/bin/python", "venv/bin/python"}) do
        if os.isfile(candidate) then
            return candidate
        end
    end

    return "python3"
end

option("python")
    set_default(default_python())
    set_showmenu(true)
    set_description("Python interpreter used for extension metadata")
option_end()

option("sanitizers")
    set_default(false)
    set_showmenu(true)
    set_description("Enable AddressSanitizer and UndefinedBehaviorSanitizer for debug builds")
option_end()

local function add_python_sysincludedirs(target)
    local seen = {}
    local dirs = {
        os.getenv("PYTHON_INCLUDE_DIR"),
        ".venv/include",
        ".venv/include/python3.9",
        ".venv/include/python3.10",
        ".venv/include/python3.11",
        ".venv/include/python3.12",
        "venv/include",
        "venv/include/python3.9",
        "venv/include/python3.10",
        "venv/include/python3.11",
        "venv/include/python3.12",
        "/usr/include/python3.9",
        "/usr/include/python3.10",
        "/usr/include/python3.11",
        "/usr/include/python3.12",
        "/usr/local/include/python3.9",
        "/usr/local/include/python3.10",
        "/usr/local/include/python3.11",
        "/usr/local/include/python3.12",
    }

    for _, dir in ipairs(dirs) do
        if dir and dir ~= "" and not seen[dir] then
            target:add("sysincludedirs", dir)
            seen[dir] = true
        end
    end
end

rule("libtorch_runtime")
    on_load(function (target)
        local torch = target:pkg("libtorch")
        local torchdir = torch:installdir()

        target:add("defines", format("TYPETORCH_LIBTORCH_ROOT=\"%s\"", torchdir))
        target:add("sysincludedirs", path.join(torchdir, "include"))
        target:add("sysincludedirs", path.join(torchdir, "include", "torch", "csrc", "api", "include"))
        target:add("linkdirs", path.join(torchdir, "lib"))
        target:add("rpathdirs", path.join(torchdir, "lib"))
        target:add("links", "torch", "torch_cpu", "c10")
    end)
rule_end()

rule("python_extension")
    on_load(function (target)
        add_python_sysincludedirs(target)
    end)

    after_load(function (target)
        target:set("extension", ".so")
    end)
rule_end()

rule("python_headers")
    on_load(function (target)
        add_python_sysincludedirs(target)
    end)
rule_end()

local function add_common_target_settings(options)
    options = options or {}

    if options.default ~= nil then
        set_default(options.default)
    end

    set_kind(options.kind or "binary")

    add_rules("libtorch_runtime", options.python_rule or "python_headers")
    add_packages("libtorch")

    if options.torch_python then
        add_links("torch_python")
    end

    -- Intentionally keep this common for all targets.
    -- This reduces compiler command-line differences for shared module files.
    add_includedirs(".", "third_party/fast_io/include")

    add_fast_io()
    add_common_mode_flags()
end

target("typetorch_cpp_debug")
    add_common_target_settings()
    add_files("examples/cpp_debug.cpp")
    add_typetorch_modules()
    add_typetorch_examples_modules()
target_end()

target("typetorch_forwarding_benchmark")
    add_common_target_settings({default = false})
    add_files("benchmarks/forwarding_benchmark.cpp")
    add_typetorch_modules()
target_end()

target("binary_size_tensor_probe")
    add_common_target_settings({default = false})
    add_files("tests/binary_size_tensor_probe.cpp")
    add_typetorch_modules()
target_end()

target("binary_size_tensor_checked_probe")
    add_common_target_settings({default = false})
    add_files("tests/binary_size_tensor_checked_probe.cpp")
    add_typetorch_modules()
target_end()

target("binary_size_tensor_unsafe_probe")
    add_common_target_settings({default = false})
    add_files("tests/binary_size_tensor_unsafe_probe.cpp")
    add_typetorch_modules()
target_end()

target("binary_size_libtorch_probe")
    add_common_target_settings({default = false})
    add_files("src/libtorch.mpp")
    add_files("tests/binary_size_libtorch_probe.cpp")
target_end()

target("typetorch_tensor_arithmetic_test")
    add_common_target_settings({default = false})
    add_files("tests/tensor_arithmetic_test.cpp")
    add_typetorch_modules()
target_end()

target("core")
    add_common_target_settings({
        kind = "static",
        torch_python = true
    })
    add_files("src/test.cpp")
    add_typetorch_modules()
target_end()

target("typetorch_capi_ext")
    add_common_target_settings({
        default = false,
        kind = "shared",
        python_rule = "python_extension",
        torch_python = true
    })

    add_files("bindings/python.mpp")
    add_files("bindings/torch_python.mpp")
    add_files("bindings/capi_bridge.mpp")
    add_files("bindings/capi_module.cpp")
    add_files("src/typetorch_capi_reflect.mpp")

    add_typetorch_modules()
    add_typetorch_examples_modules()

    set_prefixname("")
    set_basename("typetorch_capi")
target_end()