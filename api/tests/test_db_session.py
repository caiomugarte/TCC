import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]


class DatabaseSessionUrlTests(unittest.TestCase):
    def _probe(self, database_url: str, cwd: str) -> dict[str, str | None]:
        environment = os.environ.copy()
        environment["DATABASE_URL"] = database_url
        environment["PYTHONPATH"] = str(API_ROOT)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from app.db.session import DATABASE_URL, engine; "
                "print(json.dumps({'database_url': DATABASE_URL, 'database': engine.url.database}))",
            ],
            cwd=cwd,
            env=environment,
            capture_output=True,
            check=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_relative_sqlite_url_is_resolved_from_api_root_not_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as cwd:
            result = self._probe("sqlite:///./prumo-dev.db", cwd)

        expected_path = (API_ROOT / "prumo-dev.db").resolve()
        self.assertEqual(result["database"], str(expected_path))
        self.assertEqual(result["database_url"], f"sqlite:///{expected_path}")

    def test_special_and_non_sqlite_urls_are_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as cwd:
            absolute_path = Path(cwd) / "absolute.db"
            cases = (
                ("sqlite:///:memory:", ":memory:"),
                (f"sqlite:///{absolute_path}", str(absolute_path)),
                ("postgresql+psycopg://localhost/prumo", "prumo"),
            )

            for database_url, expected_database in cases:
                with self.subTest(database_url=database_url):
                    result = self._probe(database_url, cwd)
                    self.assertEqual(result["database_url"], database_url)
                    self.assertEqual(result["database"], expected_database)


if __name__ == "__main__":
    unittest.main()
