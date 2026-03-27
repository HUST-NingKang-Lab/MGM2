#!/usr/bin/env python3
"""Create a derived MGM2 JSON config for one grid-search run.

Example:
    python scripts/build_grid_config.py \
        --base src/configs/mgm2_default.json \
        --output outputs_grid/stage1/run_001/config.json \
        --set training.learning_rate=5e-5 \
        --set loss.lambda_regression=0.5
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict


def parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        current = updated
        parts = dotted_key.split('.')
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return updated


def parse_override_args(items: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'Override must be key=value: {item}')
        key, raw_value = item.split('=', 1)
        overrides[key] = parse_value(raw_value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a derived config JSON for one MGM2 grid-search run')
    parser.add_argument('--base', type=Path, required=True, help='Base config JSON path')
    parser.add_argument('--output', type=Path, required=True, help='Output config JSON path')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override in dotted.key=value form')
    args = parser.parse_args()

    with args.base.open('r', encoding='utf-8') as handle:
        config = json.load(handle)
    updated = apply_overrides(config, parse_override_args(args.overrides))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(updated, handle, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
