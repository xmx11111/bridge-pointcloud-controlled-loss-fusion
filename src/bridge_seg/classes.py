from __future__ import annotations

import re

CLASS_NAMES = ("girder", "pier", "deck")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

_FAMILY_PATTERN = re.compile(r"^([cs])-(?:f-)?bridge(\d+)")
_SEMANTIC_BRIDGE_PATTERN = re.compile(r"^bridge_(\d+)_([a-z]+)_")


def source_type_from_scene(scene_name: str) -> str:
    """Return ``virtual`` for augmented BrPCD scenes and ``real`` otherwise."""

    return "virtual" if scene_name.startswith(("c-f-", "s-f-")) else "real"


def bridge_family_from_scene(scene_name: str) -> str:
    """Map a BrPCD scene name to its source bridge family."""

    match = _FAMILY_PATTERN.match(scene_name)
    if match is not None:
        return f"{match.group(1)}-bridge{match.group(2)}"
    semantic_match = _SEMANTIC_BRIDGE_PATTERN.match(scene_name)
    if semantic_match is not None:
        return f"semanticbridge-{semantic_match.group(2)}"
    raise ValueError(f"Cannot infer bridge family from scene: {scene_name}")
