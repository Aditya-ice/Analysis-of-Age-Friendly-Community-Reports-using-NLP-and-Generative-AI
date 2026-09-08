"""Copy the committed shared fixtures into native test bundles only."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
for destination in [
    root / "ios/ElderHelpTests/Fixtures",
    root / "android/app/src/test/resources/contracts",
]:
    destination.mkdir(parents=True, exist_ok=True)
    for source in (root / "contracts/fixtures").glob("*.json"):
        (destination / source.name).write_bytes(source.read_bytes())
