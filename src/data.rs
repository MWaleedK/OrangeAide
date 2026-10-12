use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use tokenizers::AddedToken;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::tokenizer::{Tokenizer, TokenizerImpl};

pub fn train_tokenizer() -> Result<()> {
    let mut tokenizer: TokenizerImpl<
        BPE,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    > = TokenizerImpl::new(BPE::default());

    tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::new(false, true, false)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    let mut trainer: BpeTrainer = BpeTrainer::builder()
        .vocab_size(20000)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from("<PAD>", true),
            AddedToken::from("<UNK>", true),
            AddedToken::from("<BOS>", true),
            AddedToken::from("<EOS>", true),
        ])
        .build();

    let corpus_files: Vec<String> = list_text_files(Path::new("data"))?
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();

    if corpus_files.is_empty() {
        anyhow::bail!("No .txt files found in the data directory");
    }

    println!("Training tokenizer on {} file(s):", corpus_files.len());
    for file in &corpus_files {
        println!("  {file}");
    }

    tokenizer
        .train_from_files(&mut trainer, corpus_files)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    tokenizer
        .save("tokenizer/tokenizer.json", false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    println!("Tokenizer trained and saved.");

    let test_cases: [&str; 3] = ["Once upon a time", "Hello, world!", "The Verdict\nA story."];

    for test_text in test_cases {
        let encoding = tokenizer
            .encode(test_text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let decoded: String = tokenizer
            .decode(encoding.get_ids(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        println!("Input:   {test_text:?}");
        println!("Tokens:  {:?}", encoding.get_tokens());
        println!("Decoded: {decoded:?}");

        if decoded == test_text {
            println!("Round-trip test passed.");
        } else {
            println!("Round-trip test FAILED.");
        }

        println!();
    }

    Ok(())
}

pub fn batch_to_tensors(
    inputs: &[Vec<u32>],
    targets: &[Vec<u32>],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if inputs.len() != targets.len() {
        anyhow::bail!("Input and target batch sizes do not match");
    }

    if inputs.is_empty() {
        anyhow::bail!("Batch cannot be empty");
    }

    let sequence_length: usize = inputs[0].len();

    if sequence_length == 0 {
        anyhow::bail!("Sequence length cannot be zero");
    }

    for input in inputs {
        if input.len() != sequence_length {
            anyhow::bail!("Input sequences have different lengths");
        }
    }

    for target in targets {
        if target.len() != sequence_length {
            anyhow::bail!("Target sequences have different lengths");
        }
    }

    let batch_size: usize = inputs.len();

    let input_values: Vec<u32> = inputs.iter().flatten().copied().collect();
    let target_values: Vec<u32> = targets.iter().flatten().copied().collect();

    let input_tensor: Tensor =
        Tensor::new(input_values.as_slice(), device)?.reshape((batch_size, sequence_length))?;

    let target_tensor: Tensor =
        Tensor::new(target_values.as_slice(), device)?.reshape((batch_size, sequence_length))?;

    Ok((input_tensor, target_tensor))
}

pub fn create_batch(
    token_ids: &[u32],
    sequence_length: usize,
    batch_size: usize,
    start_index: usize,
) -> Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    if sequence_length == 0 || batch_size == 0 {
        return None;
    }

    let required_tokens: usize = batch_size.checked_add(sequence_length)?;

    if start_index.checked_add(required_tokens)? > token_ids.len() {
        return None;
    }

    let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut targets: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let start: usize = start_index + batch_index;

        let input: Vec<u32> = token_ids[start..start + sequence_length].to_vec();

        let target: Vec<u32> = token_ids[start + 1..start + sequence_length + 1].to_vec();

        inputs.push(input);
        targets.push(target);
    }

    Some((inputs, targets))
}

pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer: Tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    Ok(tokenizer)
}

pub fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    Ok(encoding.get_ids().to_vec())
}

pub fn load_file(path: &Path) -> Result<String> {
    if !path.is_file() {
        anyhow::bail!("Path is not a file: {}", path.display());
    }

    let extension: String = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    if extension != "txt" {
        anyhow::bail!(
            "Unsupported file type: {}. Only .txt files are supported.",
            path.display()
        );
    }

    let text: String = std::fs::read_to_string(path)?;

    if text.trim().is_empty() {
        anyhow::bail!("Text file is empty: {}", path.display());
    }

    Ok(text)
}

/// All `.txt` files directly inside `dir`, sorted by path so the
/// result is the same on every run.
pub fn list_text_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        anyhow::bail!("Path is not a directory: {}", dir.display());
    }

    let mut files: Vec<PathBuf> = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let path: PathBuf = entry?.path();

        let is_text_file: bool = path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("txt"))
                .unwrap_or(false);

        if is_text_file {
            files.push(path);
        }
    }

    files.sort();

    Ok(files)
}

/// Loads every `.txt` file in `dir` (sorted by path) and joins them
/// with a blank line between files. Empty files are skipped.
pub fn load_directory(dir: &Path) -> Result<String> {
    let mut parts: Vec<String> = Vec::new();

    for path in list_text_files(dir)? {
        let text: String = std::fs::read_to_string(&path)
            .with_context(|| format!("Could not read {}", path.display()))?;

        if !text.trim().is_empty() {
            parts.push(text);
        }
    }

    if parts.is_empty() {
        anyhow::bail!("No non-empty .txt files found in {}", dir.display());
    }

    Ok(parts.join("\n\n"))
}

pub fn create_input_target_sequences(
    token_ids: &[u32],
    sequence_length: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    let mut targets: Vec<Vec<u32>> = Vec::new();

    if sequence_length == 0 || token_ids.len() <= sequence_length {
        return (inputs, targets);
    }

    let number_of_sequences: usize = token_ids.len() - sequence_length;

    for i in 0..number_of_sequences {
        let input: Vec<u32> = token_ids[i..i + sequence_length].to_vec();

        let target: Vec<u32> = token_ids[i + 1..i + sequence_length + 1].to_vec();

        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

pub fn split_tokens(token_ids: &[u32], validation_fraction: f64) -> Result<(Vec<u32>, Vec<u32>)> {
    if !validation_fraction.is_finite() || validation_fraction <= 0.0 || validation_fraction >= 1.0
    {
        anyhow::bail!("Validation fraction must be greater than 0 and less than 1");
    }

    if token_ids.len() < 2 {
        anyhow::bail!("At least two tokens are required to split the data");
    }

    let validation_size: usize =
        ((token_ids.len() as f64 * validation_fraction).ceil() as usize).max(1);

    let training_size: usize = token_ids.len() - validation_size;

    if training_size == 0 {
        anyhow::bail!("Training data cannot be empty");
    }

    let training_tokens: Vec<u32> = token_ids[..training_size].to_vec();

    let validation_tokens: Vec<u32> = token_ids[training_size..].to_vec();

    Ok((training_tokens, validation_tokens))
}
