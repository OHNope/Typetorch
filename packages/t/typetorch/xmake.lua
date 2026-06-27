local typetorch_root = path.join(os.scriptdir(), "../../..")

includes(path.join(typetorch_root, "xmake_typetorch_modules.lua"))

local module_file_groups = {
    fast_io_module_files,
    libtorch_module_files,
    typetorch_module_files,
}

package("typetorch")
    set_kind("library", {headeronly = true})
    set_description("Type-safe LibTorch tensor wrappers with C++26 compile-time contracts")
    set_license("MIT")
    set_sourcedir(typetorch_root)

    local libtorch_variant = os.getenv("TYPETORCH_LIBTORCH_VARIANT") or "cu128"
    local libtorch_version = libtorch_variant == "cpu" and "2.8.0+cpu" or "2.8.0+cu128"
    add_deps("libtorch-bin " .. libtorch_version, {alias = "libtorch"})

    on_install(function (package)
        local moddir = package:installdir("modules")
        local incdir = package:installdir("include")

        os.cp("third_party/fast_io/include", package:installdir())
        os.cp("src/attributes.inc", incdir)
        os.cp("src/type_macros.inc", incdir)

        for _, files in ipairs(module_file_groups) do
            for _, file in ipairs(files) do
                local dst = path.join(moddir, file)
                os.mkdir(path.directory(dst))
                os.cp(file, dst)
            end
        end
    end)

    on_load(function (package)
        package:add("includedirs", "include", "modules")
        package:add("cxxflags", "-Wall", "-Wextra", "-fvisibility=hidden")
    end)
package_end()
