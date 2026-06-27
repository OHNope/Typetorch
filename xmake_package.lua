-- xmake_package.lua — Include this from consumer projects to use typetorch
-- via add_requires() + add_rules("typetorch.modules").
--
-- Usage:
--   add_repositories("typetorch-repo path/to/typetorch")
--   includes("path/to/typetorch/xmake_package.lua")
--   add_requires("typetorch")
--
--   target("my_app")
--       add_packages("typetorch")
--       add_rules("typetorch.modules")
--       add_typetorch_modules()
--       add_files("src/main.cpp")

local typetorch_root = os.scriptdir()

if os.isfile(path.join(typetorch_root, "xmake_typetorch_modules.lua")) then
    includes(path.join(typetorch_root, "xmake_typetorch_modules.lua"))
end

function add_typetorch_modules()
    for _, file in ipairs(typetorch_module_files) do
        add_files(path.join(typetorch_root, file))
    end
    for _, file in ipairs(libtorch_module_files) do
        add_files(path.join(typetorch_root, file))
    end
    for _, file in ipairs(fast_io_module_files) do
        add_files(path.join(typetorch_root, file))
    end
end

rule("typetorch.modules")
    on_load(function (target)
        target:add("cxxflags", "-fmodules", {force = true})
        target:add("cxxflags", "-freflection", {force = true})
        target:add("includedirs", path.join(typetorch_root, "src"))
        target:add("includedirs", path.join(typetorch_root, "third_party", "fast_io", "include"))
        local relocate_ldflags = os.getenv("TYPETORCH_GCC_RELOCATE_LDFLAGS")
        if relocate_ldflags and relocate_ldflags ~= "" then
            for flag in relocate_ldflags:gmatch("%S+") do
                target:add("ldflags", flag, {force = true})
                target:add("shflags", flag, {force = true})
            end
        end
    end)
    after_load(function (target)
        local pkg = target:pkg("typetorch")
        if not pkg then
            return
        end
        for _, linkdir in ipairs(pkg:get("linkdirs") or {})
        do
            if os.isfile(path.join(linkdir, "libtorch.so")) then
                target:add("rpathdirs", linkdir)
            end
        end
    end)
rule_end()
