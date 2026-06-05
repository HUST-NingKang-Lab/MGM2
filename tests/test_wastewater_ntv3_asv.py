from pathlib import Path

import numpy as np
import pandas as pd

from scripts.downstreams.wastewater.extract_mgm2_cls_cache import _resolve_plant_embeddings
from scripts.downstreams.wastewater.build_asv_ntv3_embeddings import collect_plant_asv_sequences


def test_resolve_plant_embeddings_prefers_plant_scoped_ntv3_ids():
    ntv3_embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    fallback_map = {"genus|Bacteria;Firmicutes;Bacilli;Order;Family;Genus": np.asarray([9.0, 9.0], dtype=np.float32)}
    taxonomy = pd.DataFrame(
        [
            {
                "OTU": "ASV1",
                "Kingdom": "Bacteria",
                "Phylum": "Firmicutes",
                "Class": "Bacilli",
                "Order": "Order",
                "Family": "Family",
                "Genus": "Genus",
            }
        ]
    ).set_index("OTU", drop=False)

    resolved, counts = _resolve_plant_embeddings(
        ["ASV1"],
        taxonomy,
        {"Aalborg E::ASV1": 0, "ASV1": 1},
        ntv3_embeddings,
        fallback_map,
        np.asarray([0.0, 0.0], dtype=np.float32),
        plant="Aalborg E",
    )

    assert np.allclose(resolved[0], ntv3_embeddings[0])
    assert counts == {"ntv3": 1}


def test_collect_plant_asv_sequences_uses_plant_scoped_ids(tmp_path):
    plant_dir = tmp_path / "Plant A"
    plant_dir.mkdir()
    (plant_dir / "ASVs.fasta").write_text(">ASV1\nACGT\n>ASV2\nNNaa\n")

    records = collect_plant_asv_sequences(tmp_path, plants=["Plant A"])

    assert records == [("Plant A::ASV1", "ACGT"), ("Plant A::ASV2", "NNAA")]
