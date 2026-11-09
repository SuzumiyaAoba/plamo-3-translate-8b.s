"""Pinned model profiles shared by generation, download and serving."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

MODELS = ("0.8b", "2b", "8b", "31b")
DEFAULT_MODEL = "8b"
PROFILES = Path(__file__).with_name("profiles")


@dataclass(frozen=True)
class Profile:
    size: str
    config: dict
    manifest: dict

    @property
    def name(self):
        return "plamo-3-translate-" + self.size

    @property
    def revision(self):
        return self.manifest["revision"]

    @property
    def repository(self):
        return self.manifest["repository"]

    @property
    def directory(self):
        # Preserve the existing 8B directory and keep other models separate.
        return Path("models") / ("plamo" if self.size == "8b" else self.size)

    @property
    def weights(self):
        return tuple(sorted(n for n in self.manifest["files"] if n.endswith(".safetensors")))

    def data_root(self, cache=False):
        if os.environ.get("PLAMO_DATA_DIR"):
            return Path(os.environ["PLAMO_DATA_DIR"])
        if not cache:
            return Path.cwd()
        base = Path(os.environ["XDG_DATA_HOME"]) if os.environ.get("XDG_DATA_HOME") else Path.home() / ".local/share"
        return base / self.name

    def build_directory(self, root):
        return Path(root) / "build" / ("" if self.size == "8b" else self.size)


def profile(size=DEFAULT_MODEL):
    if size not in MODELS:
        raise ValueError(f"unknown model {size!r}; choose {', '.join(MODELS)}")
    directory = PROFILES / size
    return Profile(size, json.loads((directory / "config.json").read_text()),
                   json.loads((directory / "manifest.json").read_text()))


def add_model_argument(parser):
    parser.add_argument("--model", choices=MODELS, default=os.environ.get("PLAMO_MODEL", DEFAULT_MODEL),
                        help="original BF16 checkpoint (default: PLAMO_MODEL or 8b)")
