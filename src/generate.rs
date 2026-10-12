use anyhow::Result;
use candle_core::{Device, DType, IndexOp, Tensor};

use crate::model::OrangeAideModel;

pub fn generate_greedy(
    model: &OrangeAideModel,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        anyhow::bail!("Prompt cannot be empty");
    }

    if prompt_ids
        .iter()
        .any(|&id| id >= model.vocab_size() as u32)
    {
        anyhow::bail!("Prompt contains a token ID outside the vocabulary");
    }

    if max_new_tokens == 0 {
        return Ok(prompt_ids.to_vec());
    }

    let sequence_length: usize = model.sequence_length();

    if prompt_ids.len() > sequence_length {
        anyhow::bail!(
            "Prompt length ({}) exceeds model sequence length ({})",
            prompt_ids.len(),
            sequence_length
        );
    }

    let mut generated_ids: Vec<u32> = prompt_ids.to_vec();

    for _ in 0..max_new_tokens {
        let context_start: usize =
            generated_ids.len().saturating_sub(sequence_length);

        let context: &[u32] = &generated_ids[context_start..];

        let input: Tensor =
            Tensor::new(context, device)?
                .reshape((1, context.len()))?;

        let logits: Tensor = model.forward(&input)?;

        let last_position: usize = logits.dim(1)? - 1;

        let next_token_logits: Tensor =
            logits.i((0, last_position, ..))?
                .to_dtype(DType::F32)?;

        let next_token_id: u32 =
            next_token_logits.argmax(0)?.to_scalar::<u32>()?;

        generated_ids.push(next_token_id);
    }

    Ok(generated_ids)
}

/// Advances a xorshift64 random state and returns the new value.
fn next_random(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0x9E37_79B9_7F4A_7C15;
    }

    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;

    *state
}

/// Picks the next token from raw logits.
///
/// * `temperature` <= 0 means greedy (always the highest logit).
///   Higher temperatures flatten the distribution, lower ones sharpen it.
/// * `top_k` keeps only the `top_k` highest logits before sampling.
///   Zero (or a value >= the vocabulary size) keeps all of them.
pub fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng_state: &mut u64,
) -> Result<u32> {
    if logits.is_empty() {
        anyhow::bail!("Logits cannot be empty");
    }

    if logits.iter().any(|logit| !logit.is_finite()) {
        anyhow::bail!("Logits must all be finite");
    }

    if temperature.is_nan() {
        anyhow::bail!("Temperature must be a number");
    }

    // Token indices sorted from highest to lowest logit (ties: lower index).
    let mut ranked: Vec<usize> = (0..logits.len()).collect();

    ranked.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap()
            .then(a.cmp(&b))
    });

    if temperature <= 0.0 {
        return Ok(ranked[0] as u32);
    }

    let keep: usize = if top_k == 0 || top_k >= ranked.len() {
        ranked.len()
    } else {
        top_k
    };

    ranked.truncate(keep);

    let max_logit: f32 = logits[ranked[0]];

    let weights: Vec<f64> = ranked
        .iter()
        .map(|&index| {
            (((logits[index] - max_logit) / temperature) as f64).exp()
        })
        .collect();

    let total: f64 = weights.iter().sum();

    let uniform: f64 =
        (next_random(rng_state) >> 11) as f64 / (1u64 << 53) as f64;

    let target: f64 = uniform * total;

    let mut cumulative: f64 = 0.0;

    for (position, &index) in ranked.iter().enumerate() {
        cumulative += weights[position];

        if target < cumulative {
            return Ok(index as u32);
        }
    }

    Ok(*ranked.last().unwrap() as u32)
}

/// Generates text by sampling (temperature + top-k) instead of always
/// taking the most likely token. The same seed gives the same output.
pub fn generate_sampled(
    model: &OrangeAideModel,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    seed: u64,
    device: &Device,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        anyhow::bail!("Prompt cannot be empty");
    }

    if prompt_ids
        .iter()
        .any(|&id| id >= model.vocab_size() as u32)
    {
        anyhow::bail!("Prompt contains a token ID outside the vocabulary");
    }

    if max_new_tokens == 0 {
        return Ok(prompt_ids.to_vec());
    }

    let sequence_length: usize = model.sequence_length();

    if prompt_ids.len() > sequence_length {
        anyhow::bail!(
            "Prompt length ({}) exceeds model sequence length ({})",
            prompt_ids.len(),
            sequence_length
        );
    }

    // Spread small seeds out, then discard a few outputs to warm up.
    let mut rng_state: u64 =
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_1234_9876_FEDC;

    for _ in 0..4 {
        next_random(&mut rng_state);
    }

    let mut generated_ids: Vec<u32> = prompt_ids.to_vec();

    for _ in 0..max_new_tokens {
        let context_start: usize =
            generated_ids.len().saturating_sub(sequence_length);

        let context: &[u32] = &generated_ids[context_start..];

        let input: Tensor =
            Tensor::new(context, device)?
                .reshape((1, context.len()))?;

        let logits: Tensor = model.forward(&input)?;

        let last_position: usize = logits.dim(1)? - 1;

        let next_token_logits: Vec<f32> =
            logits.i((0, last_position, ..))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;

        let next_token_id: u32 = sample_from_logits(
            &next_token_logits,
            temperature,
            top_k,
            &mut rng_state,
        )?;

        generated_ids.push(next_token_id);
    }

    Ok(generated_ids)
}