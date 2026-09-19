"""Fetch bibliographic metadata (authors, volume, pages) for the verified DOIs.

Only DOI-addressable records from OpenAlex/Crossref are used; nothing is
invented. Records that cannot be resolved are reported as unresolved.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DOIS = (
    "10.1109/CVPR46437.2021.00286",
    "10.1111/mice.13384",
    "10.1016/j.autcon.2022.104459",
    "10.1111/mice.13201",
    "10.1016/j.autcon.2021.103847",
    "10.1016/j.autcon.2021.103992",
    "10.1016/j.autcon.2022.104519",
    "10.1016/j.autcon.2023.104865",
    "10.1016/j.autcon.2023.105176",
    "10.1016/j.autcon.2023.104838",
    "10.1016/j.autcon.2025.106003",
    "10.1111/mice.13422",
    "10.1016/j.aei.2025.103373",
    "10.1016/j.autcon.2025.106336",
    "10.1016/j.autcon.2026.107243",
    "10.1016/j.dibe.2026.100912",
)

MAILTO = "codex.research@proton.me"


def _get(url: str) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": "codex-lit-check"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None


def _authors_from_crossref(item: dict) -> list[str]:
    authors: list[str] = []
    for entry in item.get("author") or []:
        family = (entry.get("family") or "").strip()
        given = (entry.get("given") or "").strip()
        literal = (entry.get("name") or "").strip()
        if family:
            initials = " ".join(
                part[0] + "." for part in given.replace("-", " ").split() if part
            )
            authors.append(f"{family}, {initials}".strip().rstrip(","))
        elif literal:
            authors.append(literal)
    return authors


def crossref(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}?mailto={MAILTO}"
    payload = _get(url)
    if not payload:
        return None
    item = payload["message"]
    date_parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    return {
        "source": "crossref",
        "doi": item.get("DOI"),
        "title": (item.get("title") or [None])[0],
        "container": (item.get("container-title") or [None])[0],
        "year": date_parts[0] if date_parts else None,
        "volume": item.get("volume"),
        "issue": item.get("issue"),
        "page": item.get("page"),
        "publisher": item.get("publisher"),
        "type": item.get("type"),
        "authors": _authors_from_crossref(item),
    }


def openalex(doi: str) -> dict | None:
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}?mailto={MAILTO}"
    payload = _get(url)
    if not payload:
        return None
    authors: list[str] = []
    for entry in payload.get("authorships") or []:
        name = (entry.get("author") or {}).get("display_name") or ""
        raw = entry.get("raw_author_name") or ""
        authors.append(raw or name)
    biblio = payload.get("biblio") or {}
    return {
        "source": "openalex",
        "doi": (payload.get("doi") or "").removeprefix("https://doi.org/"),
        "title": payload.get("title"),
        "container": ((payload.get("primary_location") or {}).get("source") or {}).get(
            "display_name"
        ),
        "year": payload.get("publication_year"),
        "volume": biblio.get("volume"),
        "issue": biblio.get("issue"),
        "page": None,
        "first_page": biblio.get("first_page"),
        "last_page": biblio.get("last_page"),
        "type": payload.get("type"),
        "authors": [a for a in authors if a],
    }


def main() -> None:
    results: dict[str, dict] = {}
    for doi in DOIS:
        record = None
        for lookup in (crossref, openalex):
            try:
                record = lookup(doi)
            except Exception as exc:  # pragma: no cover - network diagnostics
                record = {"source": lookup.__name__, "error": str(exc)}
            if record and "error" not in record:
                break
        results[doi] = record or {"error": "not found"}
        status = "OK " if record and "error" not in record else "FAIL"
        print(f"{status} {doi}")

    out = Path("work/brpcd/reference_metadata.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
