import unittest
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


class MigrationMetadataTests(unittest.TestCase):
    def test_premium_revision_is_the_current_head_after_initial_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = Config(str(root / "alembic.ini"))
        scripts = ScriptDirectory.from_config(config)

        head = scripts.get_revision("head")

        self.assertIsNotNone(head)
        self.assertEqual(head.revision, "0002_premium_foundation")
        self.assertEqual(head.down_revision, "0001_initial")


if __name__ == "__main__":
    unittest.main()
