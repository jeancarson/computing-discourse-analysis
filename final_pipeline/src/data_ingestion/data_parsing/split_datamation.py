"""Split inline section text from DataMation.json into per-section .txt files.

Behavior:
- Reads Sources/Papers/ERA_1/DataMation/DataMation.json
- Writes per-publication folders under Sources/CleanedSources/Era1/Datamation/<publication_id>/
- Writes section files named section_01.txt, section_02.txt, ... for sections with non-empty text
- Replaces `sections[].text` with `sections[].text_path`, where text_path is relative to Sources/CleanedSources (e.g. "Era1/Datamation/<pub_id>/section_01.txt")
- Writes the resulting metadata JSON to Sources/CleanedSources/Era1/Datamation/DataMationCleaned.json

This script is safe: it does not modify the original DataMation.json and will create directories as needed.
"""
from pathlib import Path
import json
import re

REPO_ROOT = Path(__file__).resolve().parents[3]

# Input file (original)
INPUT_JSON = REPO_ROOT / "data" / "original" / "ERA_1" / "DataMation" / "DataMation.json"

# Cleaned output root (user requested)
CLEANED_ROOT = REPO_ROOT / "data" / "cleaned"
# Within cleaned root the user asked for Era1/Datamation
CLEANED_SUBDIR = Path("Era1") / "Datamation"
OUTPUT_DIR = CLEANED_ROOT / CLEANED_SUBDIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "DataMationCleaned.json"

def sanitize_name(name: str) -> str:
    # keep alnum, dot, dash, underscore
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if not name:
        return "unnamed"
    return name

def main():
    if not INPUT_JSON.exists():
        print("Input JSON not found:", INPUT_JSON)
        return

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pubs = data.get("publications", [])
    for pub in pubs:
        pub_id_raw = pub.get("publication_id") or pub.get("title", "publication")
        pub_id = sanitize_name(pub_id_raw)

        pub_dir = OUTPUT_DIR / pub_id
        pub_dir.mkdir(parents=True, exist_ok=True)

        sections = pub.get("sections", [])
        for i, section in enumerate(sections, start=1):
            text = section.get("text")
            if text and isinstance(text, str) and text.strip():
                fname = f"section_{i:02}.txt"
                path = pub_dir / fname
                # write UTF-8 text file
                path.write_text(text, encoding="utf-8")
                # set relative path from Sources/CleanedSources
                rel_path = (CLEANED_SUBDIR / pub_id / fname).as_posix()
                section["text_path"] = rel_path
                section.pop("text", None)
            else:
                # Ensure we remove empty text fields to keep JSON small
                if "text" in section:
                    section.pop("text", None)

    # Save new JSON with replaced text paths
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Wrote per-section text files under:", OUTPUT_DIR)
    print("Wrote cleaned metadata JSON:", OUTPUT_JSON)

if __name__ == "__main__":
    main()
