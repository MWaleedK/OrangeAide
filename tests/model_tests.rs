use orange_aide::model::{CausalSelfAttention, MultiHeadAttention, FeedForward, LayerNorm, TransformerBlock, OrangeAideModel};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

#[test]
fn positional_embedding_preserves_shape() {
    let device: Device = Device::Cpu;

    let batch_size: usize = 2;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 16;

    let positional_embedding =
        orange_aide::model::PositionalEmbedding::new(sequence_length, embedding_dim, &device)
            .expect("Failed to create positional embedding");

    let token_embeddings: Tensor = Tensor::zeros(
        (batch_size, sequence_length, embedding_dim),
        candle_core::DType::F32,
        &device,
    )
    .expect("Failed to create token embeddings");

    let output: Tensor = positional_embedding
        .forward(&token_embeddings)
        .expect("Failed to apply positional embedding");

    assert_eq!(output.dims(), &[batch_size, sequence_length, embedding_dim]);
}

#[test]
fn qkv_projection_works_with_different_dimensions() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let embedding_dim: usize = 32;

    let var_builder = VarBuilder::zeros(DType::F32, &device);

    let projection = orange_aide::model::QkvProjection::new(embedding_dim, var_builder)
        .expect("Failed to create QKV projection");

    let input: Tensor =
        Tensor::zeros((1, 5, embedding_dim), DType::F32, &device).expect("Failed to create input");

    let (query, key, value) = projection.forward(&input).expect("QKV projection failed");

    assert_eq!(query.dims(), &[1, 5, 32]);
    assert_eq!(key.dims(), &[1, 5, 32]);
    assert_eq!(value.dims(), &[1, 5, 32]);
}

#[test]
fn qkv_projection_preserves_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let embedding_dim: usize = 16;
    let batch_size: usize = 2;
    let sequence_length: usize = 4;

    let var_builder = VarBuilder::zeros(DType::F32, &device);

    let projection = orange_aide::model::QkvProjection::new(embedding_dim, var_builder)
        .expect("Failed to create QKV projection");

    let input: Tensor = Tensor::zeros(
        (batch_size, sequence_length, embedding_dim),
        DType::F32,
        &device,
    )
    .expect("Failed to create input");

    let (query, key, value) = projection.forward(&input).expect("QKV projection failed");

    assert_eq!(query.dims(), &[batch_size, sequence_length, embedding_dim]);

    assert_eq!(key.dims(), &[batch_size, sequence_length, embedding_dim]);

    assert_eq!(value.dims(), &[batch_size, sequence_length, embedding_dim]);
}

#[test]
fn positional_embedding_rejects_long_sequence() {
    let device: Device = Device::Cpu;

    let positional_embedding = orange_aide::model::PositionalEmbedding::new(4, 16, &device)
        .expect("Failed to create positional embedding");

    let token_embeddings: Tensor = Tensor::zeros((2, 5, 16), candle_core::DType::F32, &device)
        .expect("Failed to create token embeddings");

    let result = positional_embedding.forward(&token_embeddings);

    assert!(result.is_err());
}

#[test]
fn attention_scores_have_correct_shape() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let query: Tensor =
        Tensor::zeros((2, 4, 16), DType::F32, &device).expect("Failed to create query");

    let key: Tensor = Tensor::zeros((2, 4, 16), DType::F32, &device).expect("Failed to create key");

    let scores = orange_aide::model::scaled_attention_scores(&query, &key)
        .expect("Failed to calculate attention scores");

    assert_eq!(scores.dims(), &[2, 4, 4]);
}

#[test]
fn attention_scores_scale_with_sequence_length() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let query: Tensor =
        Tensor::zeros((1, 6, 32), DType::F32, &device).expect("Failed to create query");

    let key: Tensor = Tensor::zeros((1, 6, 32), DType::F32, &device).expect("Failed to create key");

    let scores = orange_aide::model::scaled_attention_scores(&query, &key)
        .expect("Failed to calculate attention scores");

    assert_eq!(scores.dims(), &[1, 6, 6]);
}

#[test]
fn attention_scores_reject_mismatched_dimensions() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let query: Tensor =
        Tensor::zeros((1, 4, 16), DType::F32, &device).expect("Failed to create query");

    let key: Tensor = Tensor::zeros((1, 4, 32), DType::F32, &device).expect("Failed to create key");

    let result = orange_aide::model::scaled_attention_scores(&query, &key);

    assert!(result.is_err());
}

#[test]
fn causal_mask_has_correct_shape() {
    use candle_core::Device;

    let device: Device = Device::Cpu;

    let mask =
        orange_aide::model::create_causal_mask(4, &device).expect("Failed to create causal mask");

    assert_eq!(mask.dims(), &[4, 4]);
}

#[test]
fn causal_mask_blocks_future_positions() {
    use candle_core::Device;

    let device: Device = Device::Cpu;

    let mask =
        orange_aide::model::create_causal_mask(4, &device).expect("Failed to create causal mask");

    let values: Vec<Vec<f32>> = mask.to_vec2().expect("Failed to read mask");

    assert_eq!(values[0][0], 0.0);
    assert!(values[0][1].is_infinite());
    assert!(values[0][2].is_infinite());
    assert!(values[0][3].is_infinite());

    assert_eq!(values[1][0], 0.0);
    assert_eq!(values[1][1], 0.0);
    assert!(values[1][2].is_infinite());
    assert!(values[1][3].is_infinite());

    assert_eq!(values[2][0], 0.0);
    assert_eq!(values[2][1], 0.0);
    assert_eq!(values[2][2], 0.0);
    assert!(values[2][3].is_infinite());

    assert_eq!(values[3][0], 0.0);
    assert_eq!(values[3][1], 0.0);
    assert_eq!(values[3][2], 0.0);
    assert_eq!(values[3][3], 0.0);
}

#[test]
fn causal_mask_rejects_zero_length() {
    use candle_core::Device;

    let device: Device = Device::Cpu;

    let result = orange_aide::model::create_causal_mask(0, &device);

    assert!(result.is_err());
}
#[test]
fn masked_softmax_has_correct_shape() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let scores: Tensor =
        Tensor::zeros((2, 4, 4), DType::F32, &device).expect("Failed to create scores");

    let mask = orange_aide::model::create_causal_mask(4, &device).expect("Failed to create mask");

    let probabilities =
        orange_aide::model::masked_softmax(&scores, &mask).expect("Failed to calculate softmax");

    assert_eq!(probabilities.dims(), &[2, 4, 4]);
}

#[test]
fn masked_softmax_blocks_future_positions() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let scores: Tensor =
        Tensor::zeros((1, 4, 4), DType::F32, &device).expect("Failed to create scores");

    let mask = orange_aide::model::create_causal_mask(4, &device).expect("Failed to create mask");

    let probabilities =
        orange_aide::model::masked_softmax(&scores, &mask).expect("Failed to calculate softmax");

    let values: Vec<Vec<Vec<f32>>> = probabilities
        .to_vec3()
        .expect("Failed to read probabilities");

    // Token 0 can only attend to itself.
    assert!((values[0][0][0] - 1.0).abs() < 1e-6);

    // Token 1 cannot attend to tokens 2 or 3.
    assert_eq!(values[0][1][2], 0.0);
    assert_eq!(values[0][1][3], 0.0);

    // Token 2 cannot attend to token 3.
    assert_eq!(values[0][2][3], 0.0);

    // Token 3 can attend to everything.
    assert!(values[0][3][0] > 0.0);
    assert!(values[0][3][1] > 0.0);
    assert!(values[0][3][2] > 0.0);
    assert!(values[0][3][3] > 0.0);
}

#[test]
fn masked_softmax_rows_sum_to_one() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let scores: Tensor =
        Tensor::zeros((1, 4, 4), DType::F32, &device).expect("Failed to create scores");

    let mask = orange_aide::model::create_causal_mask(4, &device).expect("Failed to create mask");

    let probabilities =
        orange_aide::model::masked_softmax(&scores, &mask).expect("Failed to calculate softmax");

    let values: Vec<Vec<Vec<f32>>> = probabilities
        .to_vec3()
        .expect("Failed to read probabilities");

    for row in &values[0] {
        let sum: f32 = row.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6, "Row does not sum to 1: {}", sum);
    }
}

#[test]
fn attention_output_has_correct_shape() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let attention_weights: Tensor =
        Tensor::zeros((2, 4, 4), DType::F32, &device).expect("Failed to create attention weights");

    let value: Tensor =
        Tensor::zeros((2, 4, 16), DType::F32, &device).expect("Failed to create value");

    let output = orange_aide::model::apply_attention_weights(&attention_weights, &value)
        .expect("Failed to apply attention weights");

    assert_eq!(output.dims(), &[2, 4, 16]);
}

#[test]
fn attention_output_uses_value_vectors() {
    use candle_core::{Device, Tensor};

    let device: Device = Device::Cpu;

    let attention_weights: Tensor =
        Tensor::new(&[[[1.0f32, 0.0], [0.0, 1.0]]], &device).expect("Failed to create weights");

    let value: Tensor =
        Tensor::new(&[[[10.0f32, 20.0], [30.0, 40.0]]], &device).expect("Failed to create value");

    let output = orange_aide::model::apply_attention_weights(&attention_weights, &value)
        .expect("Failed to calculate output");

    let values: Vec<Vec<Vec<f32>>> = output.to_vec3().expect("Failed to read output");

    assert_eq!(values[0][0], vec![10.0, 20.0]);
    assert_eq!(values[0][1], vec![30.0, 40.0]);
}

#[test]
fn causal_self_attention_has_correct_output_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let batch_size: usize = 2;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 8;

    let input: Tensor = Tensor::zeros(
        (batch_size, sequence_length, embedding_dim),
        DType::F32,
        &device,
    )
    .unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: CausalSelfAttention =
        CausalSelfAttention::new(embedding_dim, var_builder).unwrap();

    let output: Tensor = attention.forward(&input).unwrap();

    assert_eq!(output.dims(), &[batch_size, sequence_length, embedding_dim]);
}

#[test]
fn causal_self_attention_handles_different_sequence_lengths() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let embedding_dim: usize = 6;

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: CausalSelfAttention =
        CausalSelfAttention::new(embedding_dim, var_builder).unwrap();

    for sequence_length in [1, 2, 5, 8] {
        let input: Tensor =
            Tensor::zeros((1, sequence_length, embedding_dim), DType::F32, &device).unwrap();

        let output: Tensor = attention.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, sequence_length, embedding_dim]);
    }
}

#[test]
fn causal_self_attention_rejects_wrong_embedding_dimension() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor = Tensor::zeros((2, 4, 7), DType::F32, &device).unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: CausalSelfAttention = CausalSelfAttention::new(8, var_builder).unwrap();

    let result = attention.forward(&input);

    assert!(result.is_err());
}

#[test]
fn causal_self_attention_rejects_wrong_input_rank() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor = Tensor::zeros((2, 8), DType::F32, &device).unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: CausalSelfAttention = CausalSelfAttention::new(8, var_builder).unwrap();

    let result = attention.forward(&input);

    assert!(result.is_err());
}

#[test]
fn causal_self_attention_zero_input_produces_zero_output() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor = Tensor::zeros((2, 4, 8), DType::F32, &device).unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: CausalSelfAttention = CausalSelfAttention::new(8, var_builder).unwrap();

    let output: Tensor = attention.forward(&input).unwrap();

    let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    for value in values {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn multi_head_attention_has_correct_output_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let batch_size: usize = 2;
    let sequence_length: usize = 5;
    let embedding_dim: usize = 8;
    let num_heads: usize = 2;

    let input: Tensor = Tensor::zeros(
        (batch_size, sequence_length, embedding_dim),
        DType::F32,
        &device,
    )
    .unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: MultiHeadAttention =
        MultiHeadAttention::new(embedding_dim, num_heads, var_builder).unwrap();

    let output: Tensor = attention.forward(&input).unwrap();

    assert_eq!(output.dims(), &[batch_size, sequence_length, embedding_dim]);
}

#[test]
fn multi_head_attention_handles_multiple_head_counts() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let embedding_dim: usize = 12;

    for num_heads in [1, 2, 3, 4, 6, 12] {
        let input: Tensor = Tensor::zeros((2, 4, embedding_dim), DType::F32, &device).unwrap();

        let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

        let attention: MultiHeadAttention =
            MultiHeadAttention::new(embedding_dim, num_heads, var_builder).unwrap();

        let output: Tensor = attention.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 4, embedding_dim]);
    }
}

#[test]
fn multi_head_attention_rejects_non_divisible_dimensions() {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let result: Result<MultiHeadAttention, _> = MultiHeadAttention::new(10, 3, var_builder);

    assert!(result.is_err());
}

#[test]
fn multi_head_attention_rejects_zero_heads() {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let result: Result<MultiHeadAttention, _> = MultiHeadAttention::new(8, 0, var_builder);

    assert!(result.is_err());
}

#[test]
fn multi_head_attention_rejects_wrong_embedding_dimension() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor = Tensor::zeros((2, 4, 7), DType::F32, &device).unwrap();

    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, &device);

    let attention: MultiHeadAttention = MultiHeadAttention::new(8, 2, var_builder).unwrap();

    let result: Result<Tensor, _> = attention.forward(&input);

    assert!(result.is_err());
}

#[test]
fn embedding_has_correct_output_shape() {
    let device: Device = Device::Cpu;

    let embedding =
        orange_aide::model::create_embedding(100, 16, &device).expect("Failed to create embedding");

    let token_ids: Tensor =
        Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &device).expect("Failed to create token tensor");

    let output: Tensor = orange_aide::model::apply_embedding(&embedding, &token_ids)
        .expect("Failed to apply embedding");

    assert_eq!(output.dims(), &[2, 3, 16]);
}

#[test]
fn feed_forward_has_correct_output_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 5, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let feed_forward: FeedForward =
        FeedForward::new(
            8,
            32,
            var_builder,
        )
        .unwrap();

    let output: Tensor =
        feed_forward.forward(&input).unwrap();

    assert_eq!(
        output.dims(),
        &[2, 5, 8]
    );
}

#[test]
fn feed_forward_handles_different_sequence_lengths() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    for sequence_length in [1, 2, 4, 8] {
        let input: Tensor =
            Tensor::zeros(
                (2, sequence_length, 8),
                DType::F32,
                &device,
            )
            .unwrap();

        let var_builder: VarBuilder =
            VarBuilder::zeros(
                DType::F32,
                &device,
            );

        let feed_forward: FeedForward =
            FeedForward::new(
                8,
                32,
                var_builder,
            )
            .unwrap();

        let output: Tensor =
            feed_forward.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[2, sequence_length, 8]
        );
    }
}

#[test]
fn feed_forward_handles_different_hidden_dimensions() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    for hidden_dim in [4, 8, 16, 32, 64] {
        let input: Tensor =
            Tensor::zeros(
                (1, 4, 8),
                DType::F32,
                &device,
            )
            .unwrap();

        let var_builder: VarBuilder =
            VarBuilder::zeros(
                DType::F32,
                &device,
            );

        let feed_forward: FeedForward =
            FeedForward::new(
                8,
                hidden_dim,
                var_builder,
            )
            .unwrap();

        let output: Tensor =
            feed_forward.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[1, 4, 8]
        );
    }
}

#[test]
fn feed_forward_rejects_zero_embedding_dimension() {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let result: Result<FeedForward, _> =
        FeedForward::new(
            0,
            32,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn feed_forward_rejects_zero_hidden_dimension() {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let result: Result<FeedForward, _> =
        FeedForward::new(
            8,
            0,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn feed_forward_rejects_wrong_embedding_dimension() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 4, 7),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let feed_forward: FeedForward =
        FeedForward::new(
            8,
            32,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        feed_forward.forward(&input);

    assert!(result.is_err());
}

#[test]
fn feed_forward_rejects_wrong_input_rank() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let feed_forward: FeedForward =
        FeedForward::new(
            8,
            32,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        feed_forward.forward(&input);

    assert!(result.is_err());
}

#[test]
fn layer_norm_has_correct_output_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 5, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let layer_norm: LayerNorm =
        LayerNorm::new(
            8,
            1e-5,
            var_builder,
        )
        .unwrap();

    let output: Tensor =
        layer_norm.forward(&input).unwrap();

    assert_eq!(
        output.dims(),
        &[2, 5, 8]
    );
}

#[test]
fn layer_norm_handles_different_sequence_lengths() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    for sequence_length in [1, 2, 4, 8] {
        let input: Tensor =
            Tensor::zeros(
                (2, sequence_length, 8),
                DType::F32,
                &device,
            )
            .unwrap();

        let var_builder: VarBuilder =
            VarBuilder::zeros(
                DType::F32,
                &device,
            );

        let layer_norm: LayerNorm =
            LayerNorm::new(
                8,
                1e-5,
                var_builder,
            )
            .unwrap();

        let output: Tensor =
            layer_norm.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[2, sequence_length, 8]
        );
    }
}


#[test]
fn layer_norm_zero_mean_and_unit_variance() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::new(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                2.0, 4.0, 6.0, 8.0,
            ],
            &device,
        )
        .unwrap()
        .reshape((2, 1, 4))
        .unwrap();

    let gamma: Tensor =
        Tensor::ones(
            4,
            DType::F32,
            &device,
        )
        .unwrap();

    let beta: Tensor =
        Tensor::zeros(
            4,
            DType::F32,
            &device,
        )
        .unwrap();

    let mut tensors: HashMap<String, Tensor> =
        HashMap::new();

    tensors.insert(
        "weight".to_string(),
        gamma,
    );

    tensors.insert(
        "bias".to_string(),
        beta,
    );

    let var_builder: VarBuilder =
        VarBuilder::from_tensors(
            tensors,
            DType::F32,
            &device,
        );

    let layer_norm: LayerNorm =
        LayerNorm::new(
            4,
            1e-5,
            var_builder,
        )
        .unwrap();

    let output: Tensor =
        layer_norm
            .forward(&input)
            .unwrap();

    let values: Vec<f32> =
        output
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    for chunk in values.chunks(4) {
        let mean: f32 =
            chunk.iter().sum::<f32>() / 4.0;

        let variance: f32 =
            chunk
                .iter()
                .map(|value| {
                    let difference: f32 =
                        *value - mean;

                    difference * difference
                })
                .sum::<f32>()
                / 4.0;

        assert!(
            mean.abs() < 1e-4,
            "Mean was {}",
            mean
        );

        assert!(
            (variance - 1.0).abs() < 1e-3,
            "Variance was {}",
            variance
        );
    }
}

#[test]
fn layer_norm_rejects_zero_embedding_dimension() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let result: Result<LayerNorm, _> =
        LayerNorm::new(
            0,
            1e-5,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn layer_norm_rejects_invalid_epsilon() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let result: Result<LayerNorm, _> =
        LayerNorm::new(
            8,
            0.0,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn layer_norm_rejects_wrong_embedding_dimension() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 4, 7),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let layer_norm: LayerNorm =
        LayerNorm::new(
            8,
            1e-5,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        layer_norm.forward(&input);

    assert!(result.is_err());
}

#[test]
fn layer_norm_rejects_wrong_input_rank() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let layer_norm: LayerNorm =
        LayerNorm::new(
            8,
            1e-5,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        layer_norm.forward(&input);

    assert!(result.is_err());
}


#[test]
fn transformer_block_has_correct_output_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 5, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let block: TransformerBlock =
        TransformerBlock::new(
            8,
            2,
            32,
            1e-5,
            var_builder,
        )
        .unwrap();

    let output: Tensor =
        block.forward(&input).unwrap();

    assert_eq!(
        output.dims(),
        &[2, 5, 8]
    );
}

#[test]
fn transformer_block_handles_different_sequence_lengths() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    for sequence_length in [1, 2, 4, 8] {
        let input: Tensor =
            Tensor::zeros(
                (2, sequence_length, 8),
                DType::F32,
                &device,
            )
            .unwrap();

        let var_builder: VarBuilder =
            VarBuilder::zeros(
                DType::F32,
                &device,
            );

        let block: TransformerBlock =
            TransformerBlock::new(
                8,
                2,
                32,
                1e-5,
                var_builder,
            )
            .unwrap();

        let output: Tensor =
            block.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[2, sequence_length, 8]
        );
    }
}

#[test]
fn transformer_block_handles_different_head_counts() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    for num_heads in [1, 2, 4, 8] {
        let input: Tensor =
            Tensor::zeros(
                (1, 4, 8),
                DType::F32,
                &device,
            )
            .unwrap();

        let var_builder: VarBuilder =
            VarBuilder::zeros(
                DType::F32,
                &device,
            );

        let block: TransformerBlock =
            TransformerBlock::new(
                8,
                num_heads,
                32,
                1e-5,
                var_builder,
            )
            .unwrap();

        let output: Tensor =
            block.forward(&input).unwrap();

        assert_eq!(
            output.dims(),
            &[1, 4, 8]
        );
    }
}

#[test]
fn transformer_block_rejects_wrong_embedding_dimension() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 4, 7),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let block: TransformerBlock =
        TransformerBlock::new(
            8,
            2,
            32,
            1e-5,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        block.forward(&input);

    assert!(result.is_err());
}

#[test]
fn transformer_block_rejects_wrong_input_rank() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::zeros(
            (2, 8),
            DType::F32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let block: TransformerBlock =
        TransformerBlock::new(
            8,
            2,
            32,
            1e-5,
            var_builder,
        )
        .unwrap();

    let result: Result<Tensor, _> =
        block.forward(&input);

    assert!(result.is_err());
}
#[test]
fn transformer_block_rejects_invalid_head_configuration() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let result: Result<TransformerBlock, _> =
        TransformerBlock::new(
            10,
            3,
            32,
            1e-5,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn transformer_block_rejects_zero_dimensions() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let result: Result<TransformerBlock, _> =
        TransformerBlock::new(
            0,
            2,
            32,
            1e-5,
            var_builder,
        );

    assert!(result.is_err());
}

#[test]
fn transformer_block_produces_finite_output() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let input: Tensor =
        Tensor::new(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                2.0, 4.0, 6.0, 8.0,
                3.0, 6.0, 9.0, 12.0,
                4.0, 8.0, 12.0, 16.0,
            ],
            &device,
        )
        .unwrap()
        .reshape((1, 4, 4))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let block: TransformerBlock =
        TransformerBlock::new(
            4,
            2,
            16,
            1e-5,
            var_builder,
        )
        .unwrap();

    let output: Tensor =
        block.forward(&input).unwrap();

    let values: Vec<f32> =
        output
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    for value in values {
        assert!(
            value.is_finite(),
            "Output contains non-finite value: {}",
            value
        );
    }
}

#[test]
fn orange_aide_model_produces_correct_logits_shape() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let vocab_size: usize = 100;
    let sequence_length: usize = 8;
    let embedding_dim: usize = 16;
    let num_heads: usize = 4;
    let hidden_dim: usize = 32;
    let num_layers: usize = 2;

    let token_ids: Tensor =
        Tensor::new(
            &[
                1u32, 2, 3, 4,
                5, 6, 7, 8,
            ],
            &device,
        )
        .unwrap()
        .reshape((1, 8))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            num_layers,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let logits: Tensor =
        model.forward(&token_ids).unwrap();

    assert_eq!(
        logits.dims(),
        &[1, 8, 100]
    );
}

#[test]
fn orange_aide_model_handles_batches() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::new(
            &[
                1u32, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            &device,
        )
        .unwrap()
        .reshape((2, 8))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let logits: Tensor =
        model.forward(&token_ids).unwrap();

    assert_eq!(
        logits.dims(),
        &[2, 8, 100]
    );
}

#[test]
fn orange_aide_model_handles_shorter_sequences() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let logits: Tensor =
        model.forward(&token_ids).unwrap();

    assert_eq!(
        logits.dims(),
        &[1, 4, 100]
    );
}

#[test]
fn orange_aide_model_rejects_sequence_too_long() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4, 5],
            &device,
        )
        .unwrap()
        .reshape((1, 5))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            4,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let result =
        model.forward(&token_ids);

    assert!(result.is_err());
}

#[test]
fn orange_aide_model_rejects_zero_length_sequence() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::zeros(
            (1, 0),
            DType::U32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let result =
        model.forward(&token_ids);

    assert!(result.is_err());
}

#[test]
fn orange_aide_model_rejects_wrong_input_rank() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::zeros(
            (8,),
            DType::U32,
            &device,
        )
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let result =
        model.forward(&token_ids);

    assert!(result.is_err());
}

#[test]
fn orange_aide_model_rejects_zero_layers() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let result =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            0,
            1e-5,
            var_builder,
            &device,
        );

    assert!(result.is_err());
}


#[test]
fn orange_aide_model_reports_configuration() {
    use candle_core::Device;
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            500,
            64,
            32,
            4,
            128,
            3,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    assert_eq!(model.vocab_size(), 500);
    assert_eq!(model.embedding_dim(), 32);
    assert_eq!(model.sequence_length(), 64);
}


#[test]
fn orange_aide_model_produces_finite_logits() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    let device: Device = Device::Cpu;

    let token_ids: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let var_builder: VarBuilder =
        VarBuilder::zeros(
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            100,
            8,
            16,
            4,
            32,
            2,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let logits: Tensor =
        model.forward(&token_ids)
            .unwrap();

    let values: Vec<f32> =
        logits
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    for value in values {
        assert!(
            value.is_finite(),
            "Non-finite logit: {}",
            value
        );
    }
}

#[test]
fn train_corpus_processes_multiple_steps() -> anyhow::Result<()> {
    use candle_core::Device;
    use candle_nn::{AdamW, VarBuilder, VarMap};

    use orange_aide::config::TrainingConfig;
    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_corpus;

    let device: Device = Device::Cpu;

    let config: TrainingConfig = TrainingConfig {
        sequence_length: 4,
        embedding_dim: 16,
        num_heads: 2,
        hidden_dim: 32,
        num_layers: 1,
        learning_rate: 0.01,
        training_steps: 20,
        batch_size: 2,
        layer_norm_eps: 1e-5,
    };

    let vocab_size: usize = 10;

    let varmap: VarMap = VarMap::new();

    let vb: VarBuilder =
        VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        );

    let model: OrangeAideModel =
    OrangeAideModel::new(
        vocab_size,
        config.sequence_length,
        config.embedding_dim,
        config.num_heads,
        config.hidden_dim,
        config.num_layers,
        config.layer_norm_eps,
        vb,
        &device,
    )?;

    let mut optimizer: AdamW =
    AdamW::new_lr(
        varmap.all_vars(),
        config.learning_rate,
    )?;

    let token_ids: Vec<u32> = vec![
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
    ];

    let losses: Vec<f32> =
        train_corpus(
            &model,
            &mut optimizer,
            &token_ids,
            config.sequence_length,
            config.batch_size,
            config.training_steps,
            &device,
        )?;

    assert_eq!(
        losses.len(),
        config.training_steps
    );

    for loss in &losses {
        assert!(loss.is_finite());
        assert!(*loss > 0.0);
    }

    Ok(())
}


#[test]
fn train_corpus_works_with_real_tokenized_text() -> anyhow::Result<()> {
    use candle_core::Device;
    use candle_nn::{AdamW, VarBuilder, VarMap};

    use orange_aide::config::TrainingConfig;
    use orange_aide::data::{
        load_file,
        load_tokenizer,
        tokenize_text,
    };
    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_corpus;

    let device: Device = Device::Cpu;

    let config: TrainingConfig = TrainingConfig {
        sequence_length: 16,
        embedding_dim: 32,
        num_heads: 4,
        hidden_dim: 64,
        num_layers: 1,
        learning_rate: 0.001,
        training_steps: 5,
        batch_size: 2,
        layer_norm_eps: 1e-5,
    };

    let text: String =
        load_file(std::path::Path::new("data/corpus.txt"))?;

    let tokenizer =
        load_tokenizer()?;

    let token_ids: Vec<u32> =
        tokenize_text(&tokenizer, &text)?;

    assert!(!token_ids.is_empty());

    let vocab_size: usize =
        tokenizer.get_vocab_size(true);

    assert!(vocab_size > 0);

    let varmap: VarMap =
        VarMap::new();

    let vb: VarBuilder =
        VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            config.sequence_length,
            config.embedding_dim,
            config.num_heads,
            config.hidden_dim,
            config.num_layers,
            config.layer_norm_eps,
            vb,
            &device,
        )?;

    let mut optimizer: AdamW =
        AdamW::new_lr(
            varmap.all_vars(),
            config.learning_rate,
        )?;

    let losses: Vec<f32> =
        train_corpus(
            &model,
            &mut optimizer,
            &token_ids,
            config.sequence_length,
            config.batch_size,
            config.training_steps,
            &device,
        )?;

    assert_eq!(
        losses.len(),
        config.training_steps
    );

    for loss in &losses {
        assert!(loss.is_finite());
        assert!(*loss > 0.0);
    }

    Ok(())
}



#[test]
fn checkpoint_load_restores_model_weights() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::{
        load_checkpoint,
        save_checkpoint,
    };

    let device: Device = Device::Cpu;

    let vocab_size: usize = 10;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 8;
    let num_heads: usize = 2;
    let hidden_dim: usize = 16;
    let num_layers: usize = 1;
    let eps: f64 = 1e-5;

    let checkpoint_path =
        std::path::Path::new(
            "checkpoints/test_checkpoint.safetensors",
        );

    // Create original model.
    let original_varmap: VarMap =
        VarMap::new();

    let original_vb: VarBuilder =
        VarBuilder::from_varmap(
            &original_varmap,
            DType::F32,
            &device,
        );

    let original_model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            num_layers,
            eps,
            original_vb,
            &device,
        )?;

    // Give the model some input.
    let input: Tensor =
        Tensor::new(
            &[
                [1u32, 2, 3, 4],
            ],
            &device,
        )?;

    let original_output: Tensor =
        original_model.forward(&input)?;

    // Save original weights.
    save_checkpoint(
        &original_varmap,
        checkpoint_path,
    )?;

    // Create a completely fresh model.
    let mut loaded_varmap: VarMap =
        VarMap::new();

    let loaded_vb: VarBuilder =
        VarBuilder::from_varmap(
            &loaded_varmap,
            DType::F32,
            &device,
        );

    let loaded_model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            num_layers,
            eps,
            loaded_vb,
            &device,
        )?;

    // Load the saved weights.
    load_checkpoint(
        &mut loaded_varmap,
        checkpoint_path,
    )?;

    let loaded_output: Tensor =
        loaded_model.forward(&input)?;

    // The outputs should now be identical.
    let difference: Tensor =
        (&original_output - &loaded_output)?
            .abs()?
            .max_all()?;

    let max_difference: f32 =
        difference.to_scalar::<f32>()?;

    assert!(
        max_difference < 1e-6,
        "Loaded model differs from original: {}",
        max_difference
    );

    // Clean up the test checkpoint.
    std::fs::remove_file(
        checkpoint_path,
    )?;

    Ok(())
}

#[test]
fn checkpoint_loaded_model_can_continue_training() -> anyhow::Result<()> {
    use candle_core::{DType, Device};
    use candle_nn::{AdamW, VarBuilder, VarMap};

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::{
        load_checkpoint,
        save_checkpoint,
        train_corpus,
    };

    let device: Device = Device::Cpu;

    let vocab_size: usize = 10;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 8;
    let num_heads: usize = 2;
    let hidden_dim: usize = 16;
    let num_layers: usize = 1;
    let eps: f64 = 1e-5;

    let token_ids: Vec<u32> = vec![
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
    ];

    let checkpoint_path =
        std::path::Path::new(
            "checkpoints/test_continue.safetensors",
        );

    // -------------------------------------------------
    // Train the first model.
    // -------------------------------------------------

    let original_varmap: VarMap =
        VarMap::new();

    let original_vb: VarBuilder =
        VarBuilder::from_varmap(
            &original_varmap,
            DType::F32,
            &device,
        );

    let original_model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            num_layers,
            eps,
            original_vb,
            &device,
        )?;

    let mut original_optimizer: AdamW =
        AdamW::new_lr(
            original_varmap.all_vars(),
            0.01,
        )?;

    let first_losses: Vec<f32> =
        train_corpus(
            &original_model,
            &mut original_optimizer,
            &token_ids,
            sequence_length,
            2,
            5,
            &device,
        )?;

    assert_eq!(
        first_losses.len(),
        5
    );

    // Save the trained model.
    save_checkpoint(
        &original_varmap,
        checkpoint_path,
    )?;

    // -------------------------------------------------
    // Create a fresh model and load the checkpoint.
    // -------------------------------------------------

    let mut loaded_varmap: VarMap =
        VarMap::new();

    let loaded_vb: VarBuilder =
        VarBuilder::from_varmap(
            &loaded_varmap,
            DType::F32,
            &device,
        );

    let loaded_model: OrangeAideModel =
        OrangeAideModel::new(
            vocab_size,
            sequence_length,
            embedding_dim,
            num_heads,
            hidden_dim,
            num_layers,
            eps,
            loaded_vb,
            &device,
        )?;

    load_checkpoint(
        &mut loaded_varmap,
        checkpoint_path,
    )?;

    // -------------------------------------------------
    // Continue training from the loaded weights.
    // -------------------------------------------------

    let mut loaded_optimizer: AdamW =
        AdamW::new_lr(
            loaded_varmap.all_vars(),
            0.01,
        )?;

    let continued_losses: Vec<f32> =
        train_corpus(
            &loaded_model,
            &mut loaded_optimizer,
            &token_ids,
            sequence_length,
            2,
            5,
            &device,
        )?;

    assert_eq!(
        continued_losses.len(),
        5
    );

    for loss in continued_losses {
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    std::fs::remove_file(
        checkpoint_path,
    )?;

    Ok(())
}






















