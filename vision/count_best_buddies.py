#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Count the number of discovered best buddies in a JSON file."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to best_buddies.json",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON to be a list, got {type(data).__name__}")

    print(f"Number of discovered best buddies: {len(data)}")


if __name__ == "__main__":
    main()