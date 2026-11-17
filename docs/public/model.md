# Model components

A request supplies a public task type t in {choice, noul, score}. The adapter receives only this type, never an example, user, or source ID. Stable candidate identities support alignment and deterministic ties. Candidate descriptions supply the actual decision content; keep identities stable when permuting records.

## Candidate isolation

Qwen computes the state/instruction prefix once. Query and candidate branches start from the same prefix attention KV, convolution, and recurrent states; each branch consumes only its own tokens. Branch positions restart from the prefix length. Gradient checkpoint recomputation reconstructs the prefix cache and public-type context.

ModernBERT uses directed attention: shared tokens attend only to shared tokens, and candidate tokens attend to shared tokens and their own candidate. Candidate positions restart at the same offset. Canonical packing and identity-based ties make equivalent permutations deterministic for the same inference settings. Score uses its complete numerical support directly.

## Shared-input adaptation and output-scale

For projections with the same input x, the shared latent A x is reused; each projection has its own output map B. ModernBERT's fused QKV and input/gate matrices use one latent per fused projection. Qwen shares A across same-input attention projections and MLP input projections.

$$
y = W_0x + (\alpha/r)BA\,\mathrm{Dropout}(x).
$$

At native attention-output and MLP-output projections, output-scale modifies the pretrained output direction:

$$
y = \mathrm{Diag}(1+g)W_0x + (\alpha/r)BA\,\mathrm{Dropout}(x).
$$

The gain vector g starts at zero. It is not a softmax temperature and does not scale the adapter residual.

## Private adaptation

Qwen keeps the shared rank-12 path intact and adds rank-2 task-type residuals at attention-output and MLP-output projections in the last eight layers:

$$
y' = y + s B_t A_t x.
$$

The extra B matrices start at zero, with a local RNG for A. ModernBERT uses shared rank 12 plus three private rank-2 blocks within each adapted projection; only the supplied type's private block is active. Original shared coordinates and scaling are preserved. ModernBERT's encoder and adapters are jointly updated. These are two explicit backbone-specific implementations, not a change of backbone name inside one layer definition.

## Objectives

For Choice and Noul, p = softmax(z) and q is the observed target distribution (one-hot for ordinary classification):

$$
\mathcal L_{\mathrm{class}} = \frac{-\sum_i q_i\log p_i + \alpha\sqrt K\left(1-\frac{\sum_i q_i p_i}{\sqrt{\sum_i p_i^2}}\right)}{1+\alpha}.
$$

The default alpha is 0.75. Setting alpha=0 gives CE. For ranking, explicit targets distribute mass across all clicked candidates.

Score's decimal trie defines a normalized distribution over every legal value. Its root uses frozen pretrained numeric-token rows, and a small trainable GRU models subsequent decimal prefixes. For a fractional label between grid values, the target linearly interpolates the two neighboring values. With p_j and q_j on increasing v_j, define F_k and Q_k as their cumulative distributions:

$$
\mathcal L_{\mathrm{CDF}}=-c\sum_{k=1}^{K-1}\frac{v_{k+1}-v_k}{v_K-v_1}\left[Q_k\log F_k+(1-Q_k)\log(1-F_k)\right].
$$

The default score loss is 0.5 CE + 0.5 CDF with c=3.0383850790577127, inherited from the research recipe. This is ordinal CDF log loss, not squared CRPS. Score decoding returns the distribution median for MAE and separately exposes the mean.

The current numeric alphabet supports nonnegative decimal values. Arbitrary negative scores require a different numeric-token scheme; source conversion must not silently clamp or relabel them.
