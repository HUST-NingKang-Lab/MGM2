# Migration from the earlier repository

MGM2 v2.0 uses `models/{scale}/model.safetensors` and a
versioned `config.json`. Its loading interface is `MGM2.from_pretrained`.

| Component | Earlier interface | MGM2 v2.0 interface |
|---|---|---|
| Checkpoint | `model.ckpt` | `model.safetensors` |
| Configuration | `MGM2Config` | Versioned `config.json` |
| Loader | Lightning module | `MGM2.from_pretrained` |
| Embedding CLI | Research extraction script | `mgm2-embed` |

Public display names are MGM2-Small, MGM2-Medium, MGM2-Large and MGM2-XLarge.
Path names remain lowercase. Keep each model generation in its own directory
with its matching configuration and loading interface.

The public package reads only the requested rows of the embedding bank and
reports fallback usage. Taxonomy matching uses the research builder's
rank-aware resolver and explicit global-fallback provenance.

## Output provenance

Embedding archives include the model scale, source-checkpoint SHA256 and
per-sample feature-resolution counts. The Hub bundle provides a SHA256 manifest
for checking downloaded files.
