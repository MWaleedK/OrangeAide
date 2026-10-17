use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarMap};
use std::path::Path;

use crate::model::OrangeAideModel;

pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor> {
    if logits.dims().len() != 3 {
        anyhow::bail!(
            "Expected logits with shape [batch, sequence, vocab], got {:?}",
            logits.dims()
        );
    }

    if targets.dims().len() != 2 {
        anyhow::bail!(
            "Expected targets with shape [batch, sequence], got {:?}",
            targets.dims()
        );
    }

    let batch_size: usize = logits.dims()[0];
    let sequence_length: usize = logits.dims()[1];
    let vocab_size: usize = logits.dims()[2];

    if targets.dims()[0] != batch_size
        || targets.dims()[1] != sequence_length
    {
        anyhow::bail!("Logits and targets dimensions do not match");
    }

    if vocab_size == 0 {
        anyhow::bail!("Vocabulary size cannot be zero");
    }

    let logits_flat: Tensor = logits.reshape((
        batch_size * sequence_length,
        vocab_size,
    ))?;

    let targets_flat: Tensor = targets.reshape(
        batch_size * sequence_length,
    )?;

    let log_probs: Tensor = candle_nn::ops::log_softmax(
        &logits_flat,
        candle_core::D::Minus1,
    )?;

    let target_log_probs: Tensor = log_probs
        .gather(&targets_flat.unsqueeze(1)?, 1)?
        .squeeze(1)?;

    let loss: Tensor = target_log_probs.neg()?.mean_all()?;

    Ok(loss)
}

pub fn train_step(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<f32> {
    let logits: Tensor = model.forward(inputs)?;
    let loss: Tensor = cross_entropy_loss(&logits, targets)?;

    let loss_value: f32 = loss.to_scalar::<f32>()?;

    optimizer.backward_step(&loss)?;

    Ok(loss_value)
}

pub fn train_model(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    inputs: &Tensor,
    targets: &Tensor,
    steps: usize,
) -> Result<Vec<f32>> {
    if steps == 0 {
        anyhow::bail!("Training steps cannot be zero");
    }

    let mut losses: Vec<f32> = Vec::with_capacity(steps);

    for _step in 0..steps {
        let loss: f32 = train_step(
            model,
            optimizer,
            inputs,
            targets,
        )?;

        losses.push(loss);
    }

    Ok(losses)
}

pub fn train_batches(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    batches: &[(Tensor, Tensor)],
) -> Result<Vec<f32>> {
    if batches.is_empty() {
        anyhow::bail!("Training batches cannot be empty");
    }

    let mut losses: Vec<f32> = Vec::with_capacity(batches.len());

    for (inputs, targets) in batches {
        let loss: f32 = train_step(
            model,
            optimizer,
            inputs,
            targets,
        )?;

        losses.push(loss);
    }

    Ok(losses)
}

pub fn train_corpus(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    token_ids: &[u32],
    sequence_length: usize,
    batch_size: usize,
    steps: usize,
    device: &Device,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        anyhow::bail!("Token IDs cannot be empty");
    }

    if sequence_length == 0 {
        anyhow::bail!("Sequence length cannot be zero");
    }

    if batch_size == 0 {
        anyhow::bail!("Batch size cannot be zero");
    }

    if steps == 0 {
        anyhow::bail!("Training steps cannot be zero");
    }

    if token_ids.len() <= sequence_length {
        anyhow::bail!(
            "Corpus must contain more tokens than the sequence length"
        );
    }

    let max_start: usize = token_ids.len() - sequence_length - 1 + 1;

    if max_start == 0 {
        anyhow::bail!("No valid training windows are available");
    }

    let mut losses: Vec<f32> = Vec::with_capacity(steps);

    // Deterministic pseudo-random sampling; no additional crate required.
    let mut rng_state: u64 = 0xA5A5_1234_9876_FEDC;

    for step in 0..steps {
        let mut input_sequences: Vec<Vec<u32>> =
            Vec::with_capacity(batch_size);

        let mut target_sequences: Vec<Vec<u32>> =
            Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Xorshift64 generates a repeatable sequence of sampling values.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let start: usize = (rng_state as usize) % max_start;

            let input: Vec<u32> =
                token_ids[start..start + sequence_length].to_vec();

            let target: Vec<u32> =
                token_ids[start + 1..start + sequence_length + 1].to_vec();

            input_sequences.push(input);
            target_sequences.push(target);
        }

        let (inputs, targets) = crate::data::batch_to_tensors(
            &input_sequences,
            &target_sequences,
            device,
        )?;

        let loss: f32 = train_step(
            model,
            optimizer,
            &inputs,
            &targets,
        )?;

        losses.push(loss);

        if (step + 1) % 100 == 0 {
            println!(
                "Step {}/{} | Loss: {:.4}",
                step + 1,
                steps,
                loss
            );
        }
    }

    Ok(losses)
}

pub fn save_checkpoint(
    varmap: &VarMap,
    path: &Path,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    varmap.save(path)?;

    Ok(())
}

pub fn load_checkpoint(
    varmap: &mut VarMap,
    path: &Path,
) -> Result<()> {
    if !path.is_file() {
        anyhow::bail!(
            "Checkpoint file does not exist: {}",
            path.display()
        );
    }

    varmap.load(path)?;

    Ok(())
}

/// Start indices of non-overlapping validation windows.
///
/// Each window needs `sequence_length` input tokens plus one extra token
/// for the final target, so a window starting at `start` is only valid
/// when `start + sequence_length + 1 <= token_count`.
pub fn validation_window_starts(
    token_count: usize,
    sequence_length: usize,
) -> Vec<usize> {
    if sequence_length == 0 || token_count <= sequence_length {
        return Vec::new();
    }

    (0..token_count - sequence_length)
        .step_by(sequence_length)
        .collect()
}

pub fn evaluate_loss(
    model: &OrangeAideModel,
    token_ids: &[u32],
    sequence_length: usize,
    batch_size: usize,
    device: &Device,
) -> Result<f32> {
    if token_ids.is_empty() {
        anyhow::bail!("Validation token IDs cannot be empty");
    }

    if sequence_length == 0 {
        anyhow::bail!("Sequence length cannot be zero");
    }

    if batch_size == 0 {
        anyhow::bail!("Batch size cannot be zero");
    }

    if token_ids.len() <= sequence_length {
        anyhow::bail!(
            "Validation corpus must contain more tokens than the sequence length"
        );
    }

    let window_starts: Vec<usize> =
        validation_window_starts(token_ids.len(), sequence_length);

    let mut total_loss: f64 = 0.0;
    let mut total_sequences: usize = 0;

    // Evaluate non-overlapping windows, one batch at a time.
    for batch_starts in window_starts.chunks(batch_size) {
        let current_batch_size: usize = batch_starts.len();

        let mut inputs: Vec<Vec<u32>> =
            Vec::with_capacity(current_batch_size);

        let mut targets: Vec<Vec<u32>> =
            Vec::with_capacity(current_batch_size);

        for &index in batch_starts {
            inputs.push(
                token_ids[index..index + sequence_length].to_vec()
            );

            targets.push(
                token_ids[index + 1..index + sequence_length + 1]
                    .to_vec()
            );
        }

        let (input_tensor, target_tensor) =
            crate::data::batch_to_tensors(&inputs, &targets, device)?;

        let logits: Tensor = model.forward(&input_tensor)?;

        let loss: Tensor =
            cross_entropy_loss(&logits, &target_tensor)?;

        let loss_value: f32 = loss.to_scalar::<f32>()?;

        if !loss_value.is_finite() {
            anyhow::bail!("Validation loss is not finite");
        }

        total_loss += loss_value as f64 * current_batch_size as f64;
        total_sequences += current_batch_size;
    }

    if total_sequences == 0 {
        anyhow::bail!("No validation sequences were evaluated");
    }

    Ok((total_loss / total_sequences as f64) as f32)
}

/// True when `step_number` (counting from 1) is a multiple of `interval`.
/// An interval of zero never fires.
pub fn is_interval_step(step_number: usize, interval: usize) -> bool {
    interval > 0 && step_number % interval == 0
}

/// Tracks the best (lowest) validation loss seen so far.
#[derive(Debug, Clone, Default)]
pub struct BestValidation {
    best: Option<f32>,
    evaluations_since_best: usize,
}

impl BestValidation {
    pub fn new() -> Self {
        Self::default()
    }

        /// Starts with a known loss (for example the loaded model's loss), so
    /// only a strictly lower loss counts as an improvement. A non-finite
    /// baseline is ignored.
    pub fn with_baseline(baseline_loss: f32) -> Self {
        Self {
            best: if baseline_loss.is_finite() {
                Some(baseline_loss)
            } else {
                None
            },
            evaluations_since_best: 0,
        }
    }

    /// Records a validation loss. Returns true only when it is finite and
    /// strictly lower than every previous loss.
    pub fn update(&mut self, loss: f32) -> bool {
        let improved: bool = loss.is_finite()
            && self.best.map_or(true, |best| loss < best);

        if improved {
            self.best = Some(loss);
            self.evaluations_since_best = 0;
        } else {
            self.evaluations_since_best += 1;
        }

        improved
    }

    pub fn best(&self) -> Option<f32> {
        self.best
    }

    pub fn evaluations_since_best(&self) -> usize {
        self.evaluations_since_best
    }

    /// True once `patience` evaluations in a row failed to improve.
    /// A patience of zero disables early stopping.
    pub fn should_stop(&self, patience: usize) -> bool {
        patience > 0 && self.evaluations_since_best >= patience
    }
}

/// Uses the first CUDA GPU when the program was built with CUDA support
/// and a GPU is available; otherwise falls back to the CPU.
pub fn best_device() -> Result<Device> {
    Ok(Device::cuda_if_available(0)?)
}

pub struct MonitorConfig<'a> {
    /// Run validation every this many steps (must be greater than zero).
    pub eval_interval: usize,
    /// Save the latest weights every this many steps (zero = only at the end).
    pub checkpoint_interval: usize,
    /// Stop after this many validations without improvement (zero = never).
    pub patience: usize,
    pub best_checkpoint: &'a Path,
    pub latest_checkpoint: &'a Path,
}

/// Learning rate with a linear warmup followed by cosine decay.
///
/// Steps are counted from 1. During the first `warmup_steps` the rate rises
/// linearly to `base_lr`; afterwards it decays along a cosine curve down to
/// `base_lr * min_lr_fraction` at the final step.
#[derive(Debug, Clone, Copy)]
pub struct LearningRateSchedule {
    pub base_lr: f64,
    pub warmup_steps: usize,
    pub min_lr_fraction: f64,
}

impl LearningRateSchedule {
    /// A schedule that always returns `learning_rate`.
    pub fn constant(learning_rate: f64) -> Self {
        Self {
            base_lr: learning_rate,
            warmup_steps: 0,
            min_lr_fraction: 1.0,
        }
    }

    pub fn at_step(&self, step_number: usize, total_steps: usize) -> f64 {
        let min_lr: f64 = self.base_lr * self.min_lr_fraction;

        if self.warmup_steps > 0 && step_number <= self.warmup_steps {
            return self.base_lr * step_number as f64
                / self.warmup_steps as f64;
        }

        let decay_steps: usize =
            total_steps.saturating_sub(self.warmup_steps).max(1);

        let progress: f64 = (step_number.saturating_sub(self.warmup_steps)
            as f64
            / decay_steps as f64)
            .clamp(0.0, 1.0);

        min_lr
            + 0.5
                * (self.base_lr - min_lr)
                * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;

    *state
}

/// Turns a seed into a well-mixed, non-zero random state.
pub fn rng_state_from_seed(seed: u64) -> u64 {
    let mut state: u64 =
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_1234_9876_FEDC;

    if state == 0 {
        state = 0xA5A5_1234_9876_FEDC;
    }

    for _ in 0..4 {
        xorshift64(&mut state);
    }

    state
}

/// Random window start positions in `0..max_start`.
pub fn sample_window_starts(
    rng_state: &mut u64,
    max_start: usize,
    count: usize,
) -> Vec<usize> {
    (0..count)
        .map(|_| (xorshift64(rng_state) as usize) % max_start)
        .collect()
}

/// Same as `train_corpus_with_schedule` with no learning-rate schedule
/// (the optimizer keeps its own rate) and a fixed sampling seed.
pub fn train_corpus_monitored(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    varmap: &VarMap,
    training_tokens: &[u32],
    validation_tokens: &[u32],
    sequence_length: usize,
    batch_size: usize,
    steps: usize,
    monitor: &MonitorConfig,
    device: &Device,
) -> Result<Vec<f32>> {
    train_corpus_with_schedule(
        model,
        optimizer,
        varmap,
        training_tokens,
        validation_tokens,
        sequence_length,
        batch_size,
        steps,
        monitor,
        None,
        0,
        device,
    )
}

/// Trains with periodic validation, best/latest checkpoints, early
/// stopping, an optional learning-rate schedule, and a seed that controls
/// which training windows are sampled (use a different seed per run).
///
/// The model's validation loss is measured before training starts. The
/// "best" checkpoint is only written when a later validation loss beats
/// it, so a run can never replace better weights with worse ones.
pub fn train_corpus_with_schedule(
    model: &OrangeAideModel,
    optimizer: &mut AdamW,
    varmap: &VarMap,
    training_tokens: &[u32],
    validation_tokens: &[u32],
    sequence_length: usize,
    batch_size: usize,
    steps: usize,
    monitor: &MonitorConfig,
    schedule: Option<LearningRateSchedule>,
    seed: u64,
    device: &Device,
) -> Result<Vec<f32>> {
    if monitor.eval_interval == 0 {
        anyhow::bail!("Evaluation interval cannot be zero");
    }

    if sequence_length == 0 {
        anyhow::bail!("Sequence length cannot be zero");
    }

    if batch_size == 0 {
        anyhow::bail!("Batch size cannot be zero");
    }

    if steps == 0 {
        anyhow::bail!("Training steps cannot be zero");
    }

    if training_tokens.len() <= sequence_length {
        anyhow::bail!(
            "Corpus must contain more tokens than the sequence length"
        );
    }

    let max_start: usize = training_tokens.len() - sequence_length;

    let starting_validation_loss: f32 = evaluate_loss(
        model,
        validation_tokens,
        sequence_length,
        batch_size,
        device,
    )?;

    println!("Starting validation loss: {:.4}", starting_validation_loss);

    let mut tracker: BestValidation =
        BestValidation::with_baseline(starting_validation_loss);

    let mut losses: Vec<f32> = Vec::with_capacity(steps);
    let mut rng_state: u64 = rng_state_from_seed(seed);

    let mut interval_loss_sum: f64 = 0.0;
    let mut interval_steps: usize = 0;

    for step in 0..steps {
        let step_number: usize = step + 1;

        if let Some(schedule) = schedule {
            optimizer.set_learning_rate(schedule.at_step(step_number, steps));
        }

        let starts: Vec<usize> =
            sample_window_starts(&mut rng_state, max_start, batch_size);

        let mut input_sequences: Vec<Vec<u32>> =
            Vec::with_capacity(batch_size);

        let mut target_sequences: Vec<Vec<u32>> =
            Vec::with_capacity(batch_size);

        for start in starts {
            input_sequences.push(
                training_tokens[start..start + sequence_length].to_vec(),
            );

            target_sequences.push(
                training_tokens[start + 1..start + sequence_length + 1]
                    .to_vec(),
            );
        }

        let (inputs, targets) = crate::data::batch_to_tensors(
            &input_sequences,
            &target_sequences,
            device,
        )?;

        let loss: f32 = train_step(model, optimizer, &inputs, &targets)?;

        if !loss.is_finite() {
            anyhow::bail!(
                "Training loss became non-finite at step {step_number}. \
                 The best checkpoint saved so far is kept. \
                 Try a lower learning rate."
            );
        }

        losses.push(loss);
        interval_loss_sum += loss as f64;
        interval_steps += 1;

        if is_interval_step(step_number, monitor.eval_interval) {
            let validation_loss: f32 = evaluate_loss(
                model,
                validation_tokens,
                sequence_length,
                batch_size,
                device,
            )?;

            let improved: bool = tracker.update(validation_loss);

            println!(
                "Step {}/{} | lr {:.6} | train loss {:.4} | validation loss {:.4}{}",
                step_number,
                steps,
                optimizer.learning_rate(),
                interval_loss_sum / interval_steps as f64,
                validation_loss,
                if improved { "  <- best so far" } else { "" }
            );

            interval_loss_sum = 0.0;
            interval_steps = 0;

            if improved {
                save_checkpoint(varmap, monitor.best_checkpoint)?;
            }

            if tracker.should_stop(monitor.patience) {
                println!(
                    "Early stopping: validation loss has not improved for {} evaluations.",
                    monitor.patience
                );
                break;
            }
        }

        if is_interval_step(step_number, monitor.checkpoint_interval) {
            save_checkpoint(varmap, monitor.latest_checkpoint)?;
        }
    }

    save_checkpoint(varmap, monitor.latest_checkpoint)?;

    Ok(losses)
}

/// GPT-2 style initialization for a freshly created model.
///
/// * Weights (including the embeddings and the output layer) are drawn
///   from a normal distribution with standard deviation 0.02.
/// * The two layers that write back into the residual stream (the
///   attention output and the second feed-forward layer) use a smaller
///   deviation, 0.02 / sqrt(2 * num_layers), so a deep model starts out
///   close to "pass the token information straight through" instead of
///   drowning it in random noise.
/// * Biases are zero. Layer-norm parameters are left unchanged.
///
/// Call this only on a model that is about to be trained from scratch,
/// never after loading a checkpoint.
pub fn apply_gpt_init(
    varmap: &VarMap,
    num_layers: usize,
    device: &Device,
) -> Result<()> {
    if num_layers == 0 {
        anyhow::bail!("Number of layers cannot be zero");
    }

    let default_std: f32 = 0.02;
    let residual_std: f32 = default_std / ((2 * num_layers) as f32).sqrt();

    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("Model variables are locked"))?;

    for (name, var) in data.iter() {
        if name.contains("norm") {
            continue;
        }

        let shape = var.shape().clone();

        let new_values: Tensor = if name.ends_with(".bias") {
            Tensor::zeros(&shape, candle_core::DType::F32, device)?
        } else {
            let std: f32 = if name.contains("attention.output")
                || name.contains("feed_forward.second")
            {
                residual_std
            } else {
                default_std
            };

            Tensor::randn(0f32, std, &shape, device)?
        };

        var.set(&new_values)?;
    }

    Ok(())
}



