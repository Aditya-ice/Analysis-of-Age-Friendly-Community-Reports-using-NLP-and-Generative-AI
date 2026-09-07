import json
from pathlib import Path

from elderhelp.main import create_app
from elderhelp.schemas import AnswerComplete


def write_openapi(path: Path = Path("openapi.json")) -> None:
    document = create_app().openapi()
    completion = AnswerComplete.model_json_schema(ref_template="#/components/schemas/{model}")
    document["components"]["schemas"].update(completion.pop("$defs", {}))
    document["components"]["schemas"]["AnswerComplete"] = completion
    path.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    write_openapi()
