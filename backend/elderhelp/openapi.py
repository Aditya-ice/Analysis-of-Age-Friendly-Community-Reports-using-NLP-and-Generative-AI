import json
from pathlib import Path

from elderhelp.main import create_app


def write_openapi(path: Path = Path("openapi.json")) -> None:
    path.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")


if __name__ == "__main__":
    write_openapi()
