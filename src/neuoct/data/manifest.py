from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class ManifestEntry:
    category: str
    object_id: str
    frame_id: int

    @property
    def object_key(self) -> str:
        return f"{self.category}/{self.object_id}"

    @property
    def relative_name(self) -> str:
        return f"{self.object_key}/{self.frame_id}"

    @classmethod
    def parse(cls, text: str, *, line_number: int | None = None) -> "ManifestEntry":
        token = text.strip().replace("\\", "/").split()[0] if text.strip() else ""
        parts = [part for part in token.split("/") if part]
        where = f" on line {line_number}" if line_number is not None else ""
        if len(parts) != 3:
            raise ValueError(
                f"Expected manifest entry 'category/object/frame_id'{where}, got {text!r}"
            )
        try:
            frame_id = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"frame_id must be an integer{where}, got {parts[2]!r}") from exc
        if frame_id < 0:
            raise ValueError(f"frame_id must be non-negative{where}, got {frame_id}")
        return cls(parts[0], parts[1], frame_id)


def read_manifest(path: str | Path, *, take: int = -1) -> list[ManifestEntry]:
    manifest_path = Path(path)
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            entries.append(ManifestEntry.parse(line, line_number=line_number))
    if take > 0:
        entries = entries[:take]
    if not entries:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return entries
