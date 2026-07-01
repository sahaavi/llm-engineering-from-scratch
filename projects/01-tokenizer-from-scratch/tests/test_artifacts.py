import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifacts import build_artifacts


class ArtifactTests(unittest.TestCase):
    def test_build_artifacts_writes_trace_metrics_and_widget_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            widget_dir = root / "widget"
            widget_dir.mkdir()

            result = build_artifacts(
                artifacts_dir=root / "artifacts",
                widget_data_path=widget_dir / "data.json",
            )

            self.assertIn("trace", result)
            self.assertIn("metrics", result)
            self.assertTrue((root / "artifacts" / "trace.json").exists())
            self.assertTrue((root / "artifacts" / "metrics.json").exists())
            self.assertTrue((widget_dir / "data.json").exists())


if __name__ == "__main__":
    unittest.main()
