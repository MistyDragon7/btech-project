#!/usr/bin/env python3
"""
Improved Droidmon extractor with:
- Timestamp retention (for sequence modeling)
- Strong reflection detection
- UNK fallbacks for missing fields
- Invalid line tracking
- Sorted output (by timestamp)
- Safe memory usage (streaming + optional buffering)

Output: JSONL with fields:
    timestamp, class, method, hooked_class, hooked_method, is_reflection
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

FILENAME_TARGET = "droidmon.log"


# --------------------------------------------------
# STEP 1: Parse line (robust)
# --------------------------------------------------


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        s = line.strip()
        if s.endswith(","):
            s = s[:-1]
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None


# --------------------------------------------------
# STEP 2: Extract fields (robust + improved reflection)
# --------------------------------------------------


def extract_fields(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not event or not isinstance(event, dict):
        return None

    cls = event.get("class") or "UNK"
    method = event.get("method") or "UNK"
    hooked_cls = event.get("hooked_class") or "UNK"
    hooked_method = event.get("hooked_method") or "UNK"
    timestamp = event.get("timestamp", 0)

    # Reflection detection
    is_reflection = event.get("type") == "reflection"

    if not is_reflection:
        if cls in [
            "java.lang.reflect.Method",
            "java.lang.reflect.Constructor",
            "java.lang.Class",
        ]:
            is_reflection = True

    return {
        "timestamp": timestamp,
        "class": cls,
        "method": method,
        "hooked_class": hooked_cls,
        "hooked_method": hooked_method,
        "is_reflection": bool(is_reflection),
    }


# --------------------------------------------------
# STEP 3: Iterate JSON objects
# --------------------------------------------------


def iter_json_objects_in_file(path: Path):
    invalid_lines = 0

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                parsed = parse_line(line)
                if parsed is None:
                    invalid_lines += 1
                    continue
                yield parsed
    except (OSError, UnicodeError) as exc:
        logging.warning("Skipping unreadable file %s: %s", path, exc)

    if invalid_lines > 0:
        logging.debug("File %s had %d invalid lines", path, invalid_lines)


# --------------------------------------------------
# STEP 4: Process all files
# --------------------------------------------------


def process_all(input_dir: Path, output_root: Path, recursive: bool):

    summary = {
        "files_seen": 0,
        "files_processed": 0,
        "input_lines": 0,
        "invalid_lines": 0,
        "events_extracted": 0,
        "events_written": 0,
    }

    if recursive:
        walker = input_dir.rglob(FILENAME_TARGET)
    else:
        walker = (p for p in input_dir.iterdir() if p.name == FILENAME_TARGET)

    for file_path in walker:
        file_path = Path(file_path)
        summary["files_seen"] += 1

        try:
            rel = file_path.relative_to(input_dir)
        except Exception:
            continue

        parts = rel.parts
        if len(parts) < 3:
            continue

        top = parts[0]
        parent_name = parts[-2]

        out_dir = output_root / top
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / parent_name

        logging.info("Processing %s -> %s", file_path, out_file)

        extracted_events = []
        input_lines = 0
        invalid_lines = 0

        for raw in file_path.open("r", encoding="utf-8", errors="replace"):
            input_lines += 1
            parsed = parse_line(raw.strip())

            if parsed is None:
                invalid_lines += 1
                continue

            extracted = extract_fields(parsed)
            if extracted:
                extracted_events.append(extracted)

        # Sort by timestamp (critical for sequence models)
        extracted_events.sort(key=lambda x: x.get("timestamp", 0))

        if extracted_events:
            with out_file.open("w", encoding="utf-8") as fh:
                for event in extracted_events:
                    fh.write(
                        json.dumps(event, separators=(",", ":"), ensure_ascii=False)
                    )
                    fh.write("\n")

            summary["files_processed"] += 1
            summary["events_written"] += len(extracted_events)

        summary["input_lines"] += input_lines
        summary["invalid_lines"] += invalid_lines
        summary["events_extracted"] += len(extracted_events)

        logging.info(
            "Done %s | lines=%d invalid=%d extracted=%d",
            file_path,
            input_lines,
            invalid_lines,
            len(extracted_events),
        )

    return summary


# --------------------------------------------------
# CLI
# --------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", default="raw_data")
    parser.add_argument("-o", "--output-root", default="extracted_data")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = process_all(input_dir, output_root, args.recursive)

    logging.info("SUMMARY: %s", summary)

    print(
        f"Processed {summary['files_processed']} files | "
        f"{summary['events_written']} events | "
        f"{summary['invalid_lines']} invalid lines"
    )


if __name__ == "__main__":
    main()
