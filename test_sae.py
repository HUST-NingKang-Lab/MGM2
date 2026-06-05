import json
from pathlib import Path
import h5py
import numpy as np
import torch
from src.models.BatchTopKSAE import BatchTopKSAE
out_dir = Path('outputs_sae/mgm2_xlarge_otu_batchtopk_grid_d4096/k64_aux0p03/')
config = json.loads((out_dir / 'config_summary.json').read_text())
ckpt = torch.load(out_dir / 'best.pt', map_location='cpu', weights_only=False)
model = BatchTopKSAE(
    input_dim=config['hidden_size'],
    dict_size=config['dict_size'],
    top_k=config['top_k'],
    input_unit_norm=True,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
rng = np.random.default_rng(123)
with h5py.File(config['activations'], 'r') as h:
    split = h['split'][:]
    token_type = h['token_type'][:]
    idx = np.flatnonzero((split == 1) & (token_type == 1))
    n = min(50000, len(idx))
    sample = np.sort(rng.choice(idx, size=n, replace=False))
    acts = h['activations'][sample].astype(np.float32)
batch_size = 1024
feature_counts = torch.zeros(config['dict_size'], dtype=torch.long)
per_sample_l0 = []
mses = []
evs = []
with torch.no_grad():
    for start in range(0, len(acts), batch_size):
        x = torch.from_numpy(acts[start:start+batch_size])
        out = model(x, update_inactive=False)
        fired = out.feature_acts > 0
        feature_counts += fired.sum(dim=0).cpu()
        per_sample_l0.append(fired.sum(dim=1).cpu())
        mses.append(torch.mean((out.sae_out - x.float()).pow(2)).item())
        evs.append(model.explained_variance(x.float(), out.sae_out).item())
per_sample_l0 = torch.cat(per_sample_l0).numpy()
counts = feature_counts.numpy()
used = counts > 0
sorted_counts = np.sort(counts[counts > 0])[::-1]
print(json.dumps({
    'sampled_val_otu': int(len(acts)),
    'checkpoint_step': int(ckpt.get('step', -1)),
    'checkpoint_metrics': ckpt.get('metrics', {}),
    'sample_val_mse': float(np.mean(mses)),
    'sample_val_explained_variance': float(np.mean(evs)),
    'features_used_at_least_once': int(used.sum()),
    'features_unused_in_sample': int((~used).sum()),
    'features_used_fraction': float(used.mean()),
    'per_sample_l0_mean': float(per_sample_l0.mean()),
    'per_sample_l0_p01': float(np.quantile(per_sample_l0, 0.01)),
    'per_sample_l0_p10': float(np.quantile(per_sample_l0, 0.10)),
    'per_sample_l0_p50': float(np.quantile(per_sample_l0, 0.50)),
    'per_sample_l0_p90': float(np.quantile(per_sample_l0, 0.90)),
    'per_sample_l0_p99': float(np.quantile(per_sample_l0, 0.99)),
    'top1_feature_fires': int(sorted_counts[0]) if len(sorted_counts) else 0,
    'top100_feature_fires_sum': int(sorted_counts[:100].sum()) if len(sorted_counts) else 0,
    'total_feature_fires': int(counts.sum()),
}, indent=2, sort_keys=True))