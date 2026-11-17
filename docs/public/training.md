# Training and checkpoints

The model configuration controls backbone, shared/private rank, output-scale, objective, and learning rates. The training configuration controls global batch, microbatch, epochs, RNG seed, clipping, optimizer schedule, and checkpoint interval.

The global batch is split across ranks without duplicating remainder rows. Each microbatch contributes its weighted loss sum divided by the complete global batch's weight sum; gradients are summed across ranks. Dynamic microbatch sizes therefore do not change example weighting. Task-specific parameters absent on a rank participate as zero gradients when another rank uses them. Gradient aggregation uses bounded flat buffers; it does not overlap communication with computation.

## Initialization

- Fresh run: pretrained base_model plus newly initialized trainable modules, including zero-initialized output-scale and private residual outputs.
- --initialize-from CHECKPOINT: load an exported model with the same model configuration, then start a new optimizer schedule. This is explicit continuation.
- --resume: restore resume.pt from the same output directory. Code hashes, data hash, configuration, and process count must match exactly. Optimizer, scheduler, sample position, and each rank's RNG are restored.

A nonempty output directory is rejected for fresh training. resume.pt is a trusted local training artifact loaded with PyTorch serialization; the portable inference export uses safetensors.

## Outputs

run.json records the full configuration, data hash, initialization source and world size. progress.json reports optimizer step, loss, gradient norm, visits, and elapsed training time. status.json records completion or a graceful pause. failure.json captures training-loop failures.

Create a file named STOP in the run directory to request a pause after the next optimizer update. The trainer writes an atomic resume checkpoint before returning. Remove STOP explicitly before resuming.

checkpoint/ contains config.json, trainables.safetensors, inventory.json, tokenizer files, and format.json. Qwen exports the trainable modules; ModernBERT exports its jointly trained encoder as well. Frozen tensors are reconstructed from the declared base model. Pin the model revision and retain the original base weights. Prediction can override a relocated base path using --base-model.

## Memory and kernels

The public path favors explicit, auditable behavior. It tokenizes complete records and uses configured microbatches; it is not a claim to reproduce the research workers' packing throughput. For long inputs, reduce micro_batch or supply more memory. No code path silently truncates a record. Score supports are complete and evaluation never uses training negative sampling.

Qwen supports optional acceleration provided by its installed Transformers/kernel environment. Record that environment for speed comparisons. A tiny CPU integration test uses unwrapped PyTorch fallback kernels; this does not change the released model equations.

LoRA+ is controlled by lora_plus_ratio: it multiplies the learning rate of B parameters in the shared adapters. The default is 1 (off); the separate additive private residual retains its explicit private learning rate. Union sampling is controlled during data preparation, not by the optimizer.
