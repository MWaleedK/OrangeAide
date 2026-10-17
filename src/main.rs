use orange_aide::{config, data, generate, model, train};

use anyhow::{Result, bail};
use candle_core::{DType, Device};
use candle_nn::{AdamW, VarBuilder, VarMap};
use std::env;
use std::path::Path;

use crate::config::TrainingConfig;
use crate::model::OrangeAideModel;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        Some("train") => {
            let fresh: bool = args.iter().any(|arg| arg == "--fresh");
            train_model(fresh)
        }
        Some("train-tokenizer") => data::train_tokenizer(),
        Some("generate") => {
            let prompt: String = args[1..]
                .iter()
                .filter(|arg| arg.as_str() != "--fresh")
                .cloned()
                .collect::<Vec<String>>()
                .join(" ");

            if prompt.trim().is_empty() {
                bail!(
                    "Please provide a prompt.\n\
                     Example: cargo run -- generate \"Once upon a time\""
                );
            }

            generate_text(&prompt)
        }
        Some(other) => {
            bail!(
                "Unknown command: {other}\n\
                 Usage:\n\
                 cargo run -- train\n\
                 cargo run -- train --fresh\n\
                 cargo run -- train-tokenizer\n\
                 cargo run -- generate \"Your prompt here\""
            );
        }
        None => {
            println!("OrangeAide");
            println!();
            println!("Usage:");
            println!("  cargo run -- train");
            println!("  cargo run -- train --fresh");
            println!("  cargo run -- train-tokenizer");
            println!("  cargo run -- generate \"Your prompt here\"");
            Ok(())
        }
    }
}

fn create_model(
    config: &TrainingConfig,
    vocab_size: usize,
    device: &Device,
    varmap: &VarMap,
) -> Result<OrangeAideModel> {
    let var_builder: VarBuilder = VarBuilder::from_varmap(varmap, DType::F32, device);

    let model: OrangeAideModel = OrangeAideModel::new(
        vocab_size,
        config.sequence_length,
        config.embedding_dim,
        config.num_heads,
        config.hidden_dim,
        config.num_layers,
        config.layer_norm_eps,
        var_builder,
        device,
    )?;

    Ok(model)
}

fn train_model(fresh: bool) -> Result<()> {
    let config: TrainingConfig = TrainingConfig::default();
    let device: Device = train::best_device()?;

    println!("OrangeAide training");
    println!(
        "Device: {}",
        if device.is_cuda() { "CUDA GPU" } else { "CPU" }
    );

    let corpus_dir: &Path = Path::new("data");
    let corpus: String = data::load_directory(corpus_dir)?;

    if corpus.trim().is_empty() {
        bail!("The training corpus is empty.");
    }

    let tokenizer = data::load_tokenizer()?;
    let token_ids: Vec<u32> = data::tokenize_text(&tokenizer, &corpus)?;

    let (training_tokens, validation_tokens) = data::split_tokens(&token_ids, 0.2)?;

    let minimum_tokens: usize = config.sequence_length + config.batch_size;

    if training_tokens.len() < minimum_tokens {
        bail!(
            "Not enough training tokens: {} tokens. Need at least {}.",
            training_tokens.len(),
            minimum_tokens
        );
    }

    if validation_tokens.len() <= config.sequence_length {
        bail!(
            "Not enough validation tokens: {} tokens. Need more than {}.",
            validation_tokens.len(),
            config.sequence_length
        );
    }

    let vocab_size: usize = tokenizer.get_vocab_size(true);
    let mut varmap: VarMap = VarMap::new();

    let model: OrangeAideModel = create_model(&config, vocab_size, &device, &varmap)?;

    let checkpoint_path: &Path = Path::new("checkpoints/orange_aide.safetensors");

    if !fresh && checkpoint_path.exists() {
        println!("Loading existing checkpoint...");
        train::load_checkpoint(&mut varmap, checkpoint_path)?;
    } else {
        train::apply_gpt_init(&varmap, config.num_layers, &device)?;
        println!("Starting with fresh GPT-style initialized weights.");
    }

    let mut optimizer: AdamW = AdamW::new_lr(varmap.all_vars(), config.learning_rate)?;

    println!("Vocabulary size: {vocab_size}");
    println!("Total tokens: {}", token_ids.len());
    println!("Training tokens: {}", training_tokens.len());
    println!("Validation tokens: {}", validation_tokens.len());
    println!("Training steps: {}", config.training_steps);
    println!("Learning rate: {}", config.learning_rate);

    std::fs::create_dir_all("checkpoints")?;

    let best_checkpoint: &Path = Path::new("checkpoints/orange_aide_best.safetensors");

    let latest_checkpoint: &Path = Path::new("checkpoints/orange_aide_latest.safetensors");

    let previous_checkpoint: &Path = Path::new("checkpoints/orange_aide_previous.safetensors");

    // Never reuse a best checkpoint left over from an earlier run.
    if best_checkpoint.exists() {
        std::fs::remove_file(best_checkpoint)?;
    }

    let monitor: train::MonitorConfig = train::MonitorConfig {
        eval_interval: 100,
        checkpoint_interval: 500,
        patience: 10,
        best_checkpoint,
        latest_checkpoint,
    };

    // Warm up briefly, then decay the learning rate down to 10% by the
    // last step. Warmup matters after resuming because the optimizer's
    // internal state starts from scratch.
    let schedule: train::LearningRateSchedule = train::LearningRateSchedule {
        base_lr: config.learning_rate,
        warmup_steps: 100.min(config.training_steps),
        min_lr_fraction: 0.1,
    };

    // A new seed every run, so resumed runs see different training windows.
    let seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(1);

    println!("Batch size: {}", config.batch_size);
    println!("Measuring starting validation loss on the full validation set...");

    let losses: Vec<f32> = train::train_corpus_with_schedule(
        &model,
        &mut optimizer,
        &varmap,
        &training_tokens,
        &validation_tokens,
        config.sequence_length,
        config.batch_size,
        config.training_steps,
        &monitor,
        Some(schedule),
        seed,
        &device,
    )?;

    if let (Some(first), Some(last)) = (losses.first(), losses.last()) {
        println!("First training batch loss: {first:.4}");
        println!("Last training batch loss:  {last:.4}");
    }

    // Replace the saved model only if this run beat it.
    if best_checkpoint.exists() {
        if checkpoint_path.exists() {
            std::fs::copy(checkpoint_path, previous_checkpoint)?;
            println!(
                "Previous weights backed up to {}",
                previous_checkpoint.display()
            );
        }

        std::fs::copy(best_checkpoint, checkpoint_path)?;
        println!(
            "Best-validation weights copied to {}",
            checkpoint_path.display()
        );
    } else if checkpoint_path.exists() {
        println!(
            "No validation improvement over the starting weights, so {} was left unchanged.",
            checkpoint_path.display()
        );
    } else {
        train::save_checkpoint(&varmap, checkpoint_path)?;
    }

    println!("Model ready at {}", checkpoint_path.display());

    Ok(())
}

fn generate_text(prompt: &str) -> Result<()> {
    let config: TrainingConfig = TrainingConfig::default();
    let device: Device = train::best_device()?;

    let checkpoint_path: &Path = Path::new("checkpoints/orange_aide.safetensors");

    if !checkpoint_path.exists() {
        bail!(
            "No trained checkpoint found at {}.\n\
             Run `cargo run -- train --fresh` first.",
            checkpoint_path.display()
        );
    }

    let tokenizer = data::load_tokenizer()?;
    let vocab_size: usize = tokenizer.get_vocab_size(true);
    let mut varmap: VarMap = VarMap::new();

    let model: OrangeAideModel = create_model(&config, vocab_size, &device, &varmap)?;

    train::load_checkpoint(&mut varmap, checkpoint_path)?;

    let prompt_ids: Vec<u32> = data::tokenize_text(&tokenizer, prompt)?;

    if prompt_ids.is_empty() {
        bail!("The prompt produced no tokens.");
    }

    if prompt_ids.len() > config.sequence_length {
        bail!(
            "The prompt produced {} tokens, but the model supports at most {}. \
             Try a shorter prompt.",
            prompt_ids.len(),
            config.sequence_length
        );
    }

    // Sampling settings: lower temperature = safer/more repetitive,
    // higher = more varied. top_k limits choices to the k likeliest tokens.
    let temperature: f32 = 0.8;
    let top_k: usize = 40;

    let seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(1);

    let generated_ids: Vec<u32> =
        generate::generate_sampled(&model, &prompt_ids, 50, temperature, top_k, seed, &device)?;

    let output: String = tokenizer
        .decode(&generated_ids, true)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;

    println!("Generated text:\n");
    println!("{output}");

    Ok(())
}
