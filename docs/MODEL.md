# Model and preprocessing contract

The backbone is a pre-LayerNorm Transformer without positional embeddings.
Static 1,536-dimensional OTU sequence embeddings are projected into the hidden
space. An abundance-conditioned MLP generates multiplicative and additive FiLM
coefficients. CLS has its own learned embedding and is not abundance-modulated.
Transformer attention dropout and both per-block LayerNorm epsilons follow the
exported architecture configuration; the latter is 1e-12.

The retained OTUs are chosen before abundance flooring. Within each sample,
quantile boundaries at 20%, 40%, 60% and 80% are calculated on the transformed
positive abundances. `searchsorted(..., side="left")` keeps boundary ties in
the lower bin. A sample with many tied low abundances can therefore have an
unequal number of OTUs in its five bins. No corpus-wide bin statistics are used.

CLS and padding use ignored target -100; valid OTUs use bins 0–4. A selected
masked abundance is replaced with -3 with probability 0.8, another valid
within-community abundance with probability 0.1, or its original value with
probability 0.1. The selected fraction is 0.15, floored to an integer with at
least one selected OTU for a nonempty sample. OTU identity stays visible.

The S1 reconstruction decoder receives `[CLS; projected static OTU identity]`.
Its focal classification loss has alpha 0.25 and gamma 2. The continuous target
is the same transformed abundance, and regression applies to bins 3 and 4.
The weighted objective is focal + 0.3 × high-bin MSE + 0.1 × semantic alignment.
Semantic alignment is symmetric multi-positive contrastive loss at temperature
0.07; equal normalized description groups are positives. Targets without a
valid text group are omitted from that loss.

`mgm2._research_model` retains the reference corruption and loss functions for
numerical verification. Its S0 branch describes a historical token-decoder
alternative. Released weights are S1: `token_bin` and `token_regression` are
retained for strict checkpoint compatibility but were frozen during S1 training.
The public forward method never uses those compatibility heads for predictions.

The release contains the exact selected tensors rather than a retrained model.
`config.json` records the original checkpoint SHA256 and selection epoch.
The optimizer was AdamW (learning rate 5e-5, weight decay 0.01), with a global
batch of 512. Over 1,000 warmup steps, the learning-rate multiplier increased
from 0.1 to 1; cosine decay then approached 0.1 over the planned 50-epoch
budget. Full validation ran after each epoch, with patience 5 for checkpoint
selection and early stopping.
Training configurations describe the experiment; they are not a portable
distributed training launcher. Full dataset preparation and distributed training
are separate reproducibility deliverables (see the benchmark guide).

SAE inputs are the final layer of the specified XLarge encoder, selecting every
valid OTU token including bin 0, and excluding CLS and padding. The SAE stores
its affine normalizer, decoder, encoder and frozen inference threshold. Its
decoder outputs normalized states; `denormalize` returns original state units.
