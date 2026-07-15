#!/usr/bin/env python3
"""Summarize conservative V2 token ASV-delta runs."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

from summarize_token_asv_delta_results import main as summarize_main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--result_root",
                "outputs_downstreams/wastewater_v2/token_asv_delta_transformer_optimized/conservative",
            ]
        )
    summarize_main()
