from pathlib import Path
import unittest

from common import unified_protocol as up


ROOT = Path(__file__).resolve().parents[1]
LEAVES = {
    "unet_r50", "unetpp_r50", "deeplabv3plus_r50", "segformer_b2_in1k", "segformer_b0_in1k",
    "swin_upernet", "segmenter_s16_in1k", "mobilenetv3_lraspp_in1k", "pidnet_s_in1k", "ddrnet",
    "yolov3_in1k",
}
# Names of the protocols and tools this repository deliberately leaves out.
ABSENT = ("config_oem_loveda", "config_native", "native_protocol", "config_v4_recipe",
          "evaluate_native", "optuna_aug", "model_settings", "OEM_LOVEDA_COMMON_V3")


class RepositorySurfaceTest(unittest.TestCase):
    def test_eleven_leaves(self):
        self.assertEqual(set(up.MODEL_CONFIGS), LEAVES)

    def test_only_the_unified_configs_are_shipped(self):
        expected = {"config_unified_v4.py", *(f"{name}.py" for name in up.MODEL_CONFIGS.values())}
        self.assertEqual({path.name for path in (ROOT / "configs").glob("*.py")}, expected)

    def test_only_the_final_taxonomy_is_shipped(self):
        actual = {path.name for path in (ROOT / "configs" / "class_maps").glob("*.json")}
        self.assertEqual(actual, {"unified_fine_v2.json"})

    def test_tools(self):
        self.assertEqual({path.name for path in (ROOT / "tools").glob("*.py")},
                         {"evaluate_unified.py", "fetch_pretrained.py",
                          "prepare_native_dataset.py", "run_baselines.py"})

    def test_code_does_not_name_the_removed_protocols(self):
        for folder in ("common", "configs", "tools"):
            for path in (ROOT / folder).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for name in ABSENT:
                    self.assertNotIn(name, text, f"{path}: {name}")


if __name__ == "__main__":
    unittest.main()
