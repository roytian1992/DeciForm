# Evaluation

predict writes one aligned probability vector for every request, including the complete candidate identities and selected semantic value. It uses deterministic semantic tie-breaking. Score returns the median-grid decision and expected_score separately. Default batch size is one; select larger batches only when complete inputs fit.

evaluate requires exact record-ID coverage, candidate alignment, valid probabilities, and native evaluation candidates. It rejects training views. Choice/Noul report accuracy; Score reports MAE. Multi-positive ranking reports impression-averaged AUC, all-click reciprocal-rank mean, and nDCG@5/@10. MIND MRR averages 1/rank over every clicked item in an impression, then averages impressions.

To test permutation consistency, reverse or otherwise permute the aligned options/option_ids/option_definitions arrays (and json_keys when present), keeping identities and labels intact. For Noul, reverse candidate_values. Keep the record ID unchanged. Predict again, then run:

~~~bash
python -m deciform.core permutation-audit \
  --original runs/example/original.jsonl \
  --permuted runs/example/reversed.jsonl \
  --output runs/example/permutation.json
~~~

The audit reports flips and maximum identity-aligned probability differences per task. It compares complete supplied populations; do not compare only successfully parsed or conveniently short examples. A subset benchmark should declare its selection before inference.

Architecture tests also change one candidate's text and check that other candidates' raw logits are unaffected. Their normalized probabilities can change because the final softmax denominator includes the changed candidate.
