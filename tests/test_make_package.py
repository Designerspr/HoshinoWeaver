import unittest

import make_package


class TestGeneratedSpec(unittest.TestCase):
    """Guard the PyInstaller spec generator's native-artifact collection.

    The spec is generated source, so a template mistake only surfaces during a
    real packaging run. These checks keep that failure at test time instead.
    """

    def setUp(self) -> None:
        self.spec = make_package._generate_spec(
            "/work",
            "/compile",
            "HoshinoWeaver",
            "/work/icon.ico",
        )

    def test_generated_spec_is_valid_python(self) -> None:
        compile(self.spec, "<generated.spec>", "exec")

    def test_spec_collects_compiled_extension(self) -> None:
        self.assertIn("hoshicore._custom_op._C", self.spec)

    def test_spec_collects_metal_extension_and_shader_library(self) -> None:
        self.assertIn("hoshicore._custom_op._metal", self.spec)
        self.assertIn("_metal_kernels.metallib", self.spec)
        # The runtime locates the shader library next to the loaded extension,
        # so the data destination must be the package dir, not the bundle root.
        self.assertIn("shared_datas.append((_metallib, 'hoshicore/_custom_op'))",
                      self.spec)

    def test_spec_fails_loudly_when_shader_library_is_missing(self) -> None:
        self.assertIn("if not _osp_metal.exists(_metallib):", self.spec)
        self.assertIn("raise RuntimeError(", self.spec)
        self.assertIn("is missing; rebuild with", self.spec)

    def test_metal_collection_is_conditional(self) -> None:
        # Non-macOS builds have no _metal module; collection must stay guarded
        # so Linux/Windows packaging is unaffected.
        self.assertIn(
            "_metal_spec = _imputil.find_spec('hoshicore._custom_op._metal')",
            self.spec,
        )
        self.assertIn("if _metal_spec is not None and _metal_spec.origin:", self.spec)


if __name__ == "__main__":
    unittest.main()
