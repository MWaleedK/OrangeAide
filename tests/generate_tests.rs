
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use orange_aide::generate::generate_greedy;
use orange_aide::model::OrangeAideModel;

fn create_test_model(
    device: &Device,
    vocab_size: usize,
    sequence_length: usize,
) -> Result<OrangeAideModel> {
    let varmap: VarMap = VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &varmap,
            DType::F32,
            device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            8,
            2,
            16,
            1,
            1e-5,
            var_builder,
            device,
        )?;

    Ok(model)
}

#[test]
fn greedy_generation_rejects_empty_prompt() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 8)?;

    let result: Result<Vec<u32>> =
        generate_greedy(&model, &[], 3, &device);

    assert!(result.is_err());

    Ok(())
}

#[test]
fn greedy_generation_zero_tokens_preserves_prompt() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let generated: Vec<u32> =
        generate_greedy(
            &model,
            &prompt_ids,
            0,
            &device,
        )?;

    assert_eq!(generated, prompt_ids);

    Ok(())
}

#[test]
fn greedy_generation_adds_exact_number_of_tokens() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let generated: Vec<u32> =
        generate_greedy(
            &model,
            &prompt_ids,
            4,
            &device,
        )?;

    assert_eq!(
        generated.len(),
        prompt_ids.len() + 4
    );

    assert_eq!(
        &generated[..prompt_ids.len()],
        &prompt_ids
    );

    Ok(())
}

#[test]
fn greedy_generation_rejects_prompt_longer_than_context() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 4)?;

    let prompt_ids: Vec<u32> =
        vec![1, 2, 3, 4, 5];

    let result: Result<Vec<u32>> =
        generate_greedy(
            &model,
            &prompt_ids,
            1,
            &device,
        );

    assert!(result.is_err());

    Ok(())
}

#[test]
fn greedy_generation_rejects_out_of_range_token_ids() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 8)?;

    // Valid token IDs are 0 through 19.
    let prompt_ids: Vec<u32> = vec![1, 20];

    let result: Result<Vec<u32>> =
        generate_greedy(
            &model,
            &prompt_ids,
            1,
            &device,
        );

    assert!(
        result.is_err(),
        "Generation should reject token IDs outside the vocabulary"
    );

    Ok(())
}

#[test]
fn greedy_generation_is_deterministic() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let first: Vec<u32> =
        generate_greedy(
            &model,
            &prompt_ids,
            4,
            &device,
        )?;

    let second: Vec<u32> =
        generate_greedy(
            &model,
            &prompt_ids,
            4,
            &device,
        )?;

    assert_eq!(first, second);

    Ok(())
}

#[test]
fn greedy_generation_outputs_valid_token_ids() -> Result<()> {
    let device: Device = Device::Cpu;
    let vocab_size: usize = 20;

    let model: OrangeAideModel =
        create_test_model(&device, vocab_size, 8)?;

    let generated: Vec<u32> =
        generate_greedy(
            &model,
            &[1, 2, 3],
            5,
            &device,
        )?;

    assert!(
        generated
            .iter()
            .all(|&id| (id as usize) < vocab_size),
        "Generation produced a token ID outside the vocabulary"
    );

    Ok(())
}

#[test]
fn greedy_generation_handles_full_context_window() -> Result<()> {
    let device: Device = Device::Cpu;
    let model: OrangeAideModel =
        create_test_model(&device, 20, 4)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3, 4];

    // Generate more tokens than the context window can hold.
    let generated: Vec<u32> =
        generate_greedy(
            &model,
            &prompt_ids,
            6,
            &device,
        )?;

    assert_eq!(
        generated.len(),
        prompt_ids.len() + 6
    );

    assert!(
        generated.iter().all(|&id| id < 20),
        "Generation produced a token ID outside the vocabulary"
    );

    Ok(())
}
#[test]
fn sampling_with_top_k_one_always_picks_the_highest_logit() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![0.1, 2.0, 0.5, 1.0];
    let mut rng: u64 = 12345;

    for _ in 0..20 {
        let token: u32 = sample_from_logits(&logits, 1.0, 1, &mut rng).unwrap();
        assert_eq!(token, 1);
    }
}

#[test]
fn sampling_with_zero_temperature_is_greedy() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![0.1, 2.0, 0.5, 1.0];
    let mut rng: u64 = 12345;

    for _ in 0..20 {
        let token: u32 = sample_from_logits(&logits, 0.0, 0, &mut rng).unwrap();
        assert_eq!(token, 1);
    }
}

#[test]
fn sampling_only_returns_tokens_inside_top_k() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let mut rng: u64 = 987654321;
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for _ in 0..500 {
        let token: u32 = sample_from_logits(&logits, 2.0, 3, &mut rng).unwrap();
        assert!(token < 3, "token {token} is outside the top 3");
        seen.insert(token);
    }

    assert!(seen.len() >= 2, "sampling never varied");
}

#[test]
fn sampling_prefers_higher_logits() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![0.0, 5.0];
    let mut rng: u64 = 424242;
    let mut high_count: usize = 0;

    for _ in 0..1000 {
        if sample_from_logits(&logits, 1.0, 0, &mut rng).unwrap() == 1 {
            high_count += 1;
        }
    }

    assert!(high_count > 900, "high logit chosen only {high_count}/1000 times");
}

#[test]
fn high_temperature_flattens_the_distribution() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![0.0, 5.0];
    let mut rng: u64 = 424242;
    let mut low_count: usize = 0;

    for _ in 0..1000 {
        if sample_from_logits(&logits, 50.0, 0, &mut rng).unwrap() == 0 {
            low_count += 1;
        }
    }

    assert!(low_count > 300, "low logit chosen only {low_count}/1000 times");
}

#[test]
fn sampling_is_deterministic_for_the_same_rng_state() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![1.0, 1.1, 0.9, 1.05, 0.95];

    let mut rng_a: u64 = 777;
    let mut rng_b: u64 = 777;

    let a: Vec<u32> = (0..50)
        .map(|_| sample_from_logits(&logits, 1.0, 0, &mut rng_a).unwrap())
        .collect();

    let b: Vec<u32> = (0..50)
        .map(|_| sample_from_logits(&logits, 1.0, 0, &mut rng_b).unwrap())
        .collect();

    assert_eq!(a, b);
}

#[test]
fn sampling_works_with_a_zero_rng_state() {
    use orange_aide::generate::sample_from_logits;

    let logits: Vec<f32> = vec![1.0; 10];
    let mut rng: u64 = 0;

    let tokens: Vec<u32> = (0..50)
        .map(|_| sample_from_logits(&logits, 1.0, 0, &mut rng).unwrap())
        .collect();

    assert!(tokens.iter().any(|&t| t != tokens[0]), "rng got stuck");
}

#[test]
fn sampling_rejects_empty_and_non_finite_logits() {
    use orange_aide::generate::sample_from_logits;

    let mut rng: u64 = 1;

    assert!(sample_from_logits(&[], 1.0, 0, &mut rng).is_err());
    assert!(sample_from_logits(&[1.0, f32::NAN], 1.0, 0, &mut rng).is_err());
    assert!(sample_from_logits(&[1.0, f32::INFINITY], 1.0, 0, &mut rng).is_err());
    assert!(sample_from_logits(&[1.0, 2.0], f32::NAN, 0, &mut rng).is_err());
}

#[test]
fn sampled_generation_adds_exact_number_of_tokens() -> Result<()> {
    use orange_aide::generate::generate_sampled;

    let device: Device = Device::Cpu;
    let model: OrangeAideModel = create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let generated: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 6, 1.0, 5, 42, &device)?;

    assert_eq!(generated.len(), prompt_ids.len() + 6);
    assert_eq!(&generated[..prompt_ids.len()], &prompt_ids);
    assert!(generated.iter().all(|&id| id < 20));

    Ok(())
}

#[test]
fn sampled_generation_is_deterministic_for_the_same_seed() -> Result<()> {
    use orange_aide::generate::generate_sampled;

    let device: Device = Device::Cpu;
    let model: OrangeAideModel = create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let a: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 20, 1.0, 0, 99, &device)?;
    let b: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 20, 1.0, 0, 99, &device)?;

    assert_eq!(a, b);

    Ok(())
}

#[test]
fn sampled_generation_differs_for_different_seeds() -> Result<()> {
    use orange_aide::generate::generate_sampled;

    let device: Device = Device::Cpu;
    let model: OrangeAideModel = create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let a: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 30, 1.0, 0, 1, &device)?;
    let b: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 30, 1.0, 0, 2, &device)?;

    assert_ne!(a, b);

    Ok(())
}

#[test]
fn sampled_generation_with_zero_temperature_matches_greedy() -> Result<()> {
    use orange_aide::generate::generate_sampled;

    let device: Device = Device::Cpu;
    let model: OrangeAideModel = create_test_model(&device, 20, 8)?;

    let prompt_ids: Vec<u32> = vec![1, 2, 3];

    let greedy: Vec<u32> = generate_greedy(&model, &prompt_ids, 10, &device)?;
    let sampled: Vec<u32> =
        generate_sampled(&model, &prompt_ids, 10, 0.0, 0, 7, &device)?;

    assert_eq!(greedy, sampled);

    Ok(())
}

#[test]
fn sampled_generation_rejects_bad_prompts() -> Result<()> {
    use orange_aide::generate::generate_sampled;

    let device: Device = Device::Cpu;
    let model: OrangeAideModel = create_test_model(&device, 20, 8)?;

    assert!(generate_sampled(&model, &[], 3, 1.0, 0, 1, &device).is_err());
    assert!(generate_sampled(&model, &[99], 3, 1.0, 0, 1, &device).is_err());
    assert!(
        generate_sampled(&model, &[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 1.0, 0, 1, &device)
            .is_err()
    );

    let unchanged: Vec<u32> =
        generate_sampled(&model, &[1, 2], 0, 1.0, 0, 1, &device)?;
    assert_eq!(unchanged, vec![1, 2]);

    Ok(())
}






