"""Dataset preparation tests, run with python -m unittest test_prepare_neu."""
import tempfile
import unittest
from pathlib import Path
from prepare_neu import CLASSES, prepare


class PrepareNEUTests(unittest.TestCase):
    def test_missing_images_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; source.mkdir()
            out = Path(tmp) / "out"
            with self.assertRaises(ValueError):
                prepare(source, out)
            self.assertFalse(out.exists())

    def test_maps_all_classes_and_will_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; source.mkdir()
            for prefix in CLASSES:
                for i in range(1, 301):
                    (source / f"{prefix}_{i}.bmp").write_bytes(b"fixture")
            out = Path(tmp) / "out"
            self.assertEqual(prepare(source, out), {name: 300 for name in CLASSES.values()})
            self.assertEqual(len(list(out.glob("*/*.bmp"))), 1800)
            self.assertEqual((out / "crazing/Cr_1.bmp").read_bytes(), b"fixture")
            with self.assertRaises(ValueError):
                prepare(source, out)


if __name__ == "__main__":
    unittest.main()
