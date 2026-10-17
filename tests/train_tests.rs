use orange_aide::train::cross_entropy_loss;

#[test]
fn cross_entropy_loss_returns_scalar() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let logits: Tensor = Tensor::new(&[2.0f32, 1.0, 0.0, 0.0, 1.0, 2.0], &device)
        .unwrap()
        .reshape((1, 2, 3))
        .unwrap();

    let targets: Tensor = Tensor::new(&[0u32, 2], &device)
        .unwrap()
        .reshape((1, 2))
        .unwrap();

    let loss: Tensor = cross_entropy_loss(&logits, &targets).unwrap();

    assert_eq!(loss.dims(), &[] as &[usize]);
}

#[test]
fn cross_entropy_loss_rewards_correct_predictions() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let good_logits: Tensor = Tensor::new(&[10.0f32, 0.0, 0.0], &device)
        .unwrap()
        .reshape((1, 1, 3))
        .unwrap();

    let bad_logits: Tensor = Tensor::new(&[0.0f32, 10.0, 0.0], &device)
        .unwrap()
        .reshape((1, 1, 3))
        .unwrap();

    let targets: Tensor = Tensor::new(&[0u32], &device)
        .unwrap()
        .reshape((1, 1))
        .unwrap();

    let good_loss: f32 = cross_entropy_loss(&good_logits, &targets)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let bad_loss: f32 = cross_entropy_loss(&bad_logits, &targets)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        good_loss < bad_loss,
        "Good loss {} should be lower than bad loss {}",
        good_loss,
        bad_loss
    );
}

#[test]
fn cross_entropy_loss_handles_batches() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let logits: Tensor = Tensor::new(
        &[
            5.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0,
        ],
        &device,
    )
    .unwrap()
    .reshape((2, 2, 3))
    .unwrap();

    let targets: Tensor = Tensor::new(&[0u32, 1, 2, 0], &device)
        .unwrap()
        .reshape((2, 2))
        .unwrap();

    let loss: Tensor = cross_entropy_loss(&logits, &targets).unwrap();

    let loss_value: f32 = loss.to_scalar::<f32>().unwrap();

    assert!(loss_value.is_finite());
    assert!(loss_value > 0.0);
}

#[test]
fn cross_entropy_loss_rejects_wrong_logit_rank() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let logits: Tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();

    let targets: Tensor = Tensor::zeros((2, 3), DType::U32, &device).unwrap();

    let result = cross_entropy_loss(&logits, &targets);

    assert!(result.is_err());
}

#[test]
fn cross_entropy_loss_rejects_wrong_target_rank() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let logits: Tensor = Tensor::zeros((2, 3, 10), DType::F32, &device).unwrap();

    let targets: Tensor = Tensor::zeros((2, 3, 1), DType::U32, &device).unwrap();

    let result = cross_entropy_loss(&logits, &targets);

    assert!(result.is_err());
}

#[test]
fn cross_entropy_loss_rejects_dimension_mismatch() {
    use candle_core::{DType, Device, Tensor};

    let device: Device = Device::Cpu;

    let logits: Tensor = Tensor::zeros((2, 4, 10), DType::F32, &device).unwrap();

    let targets: Tensor = Tensor::zeros((2, 3), DType::U32, &device).unwrap();

    let result = cross_entropy_loss(&logits, &targets);

    assert!(result.is_err());
}

#[test]
fn orange_aide_model_produces_valid_nonzero_gradients() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::cross_entropy_loss;

    let device: Device = Device::Cpu;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            20,
            4,
            8,
            2,
            16,
            1,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let input: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let targets: Tensor =
        Tensor::new(
            &[2u32, 3, 4, 5],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let logits: Tensor =
        model.forward(&input)
            .unwrap();

    let loss: Tensor =
        cross_entropy_loss(
            &logits,
            &targets,
        )
        .unwrap();

    let gradients =
        loss.backward()
            .unwrap();

    let variables =
        var_map.all_vars();

    assert!(
        !variables.is_empty(),
        "Model should contain trainable parameters"
    );

    let mut gradients_found: usize =
        0;

    let mut nonzero_gradients: usize =
        0;

    for variable in &variables {
        let gradient = gradients
            .get(variable.as_tensor());

        if let Some(gradient) = gradient {
            gradients_found += 1;

            assert_eq!(
                gradient.dims(),
                variable.dims(),
                "Gradient shape must match parameter shape"
            );

            let gradient_values: Vec<f32> =
                gradient
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();

            for value in &gradient_values {
                assert!(
                    value.is_finite(),
                    "Gradient contains non-finite value: {}",
                    value
                );

                if *value != 0.0 {
                    nonzero_gradients += 1;
                }
            }
        }
    }

    assert!(
        gradients_found > 0,
        "No parameter gradients were produced"
    );

    assert!(
        nonzero_gradients > 0,
        "All parameter gradients are zero"
    );
}

#[test]
fn adamw_updates_model_parameters() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{AdamW, ParamsAdamW, Optimizer, VarBuilder, VarMap};

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::cross_entropy_loss;

    let device: Device =
        Device::Cpu;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            20,
            4,
            8,
            2,
            16,
            1,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let input: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let targets: Tensor =
        Tensor::new(
            &[2u32, 3, 4, 5],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let variables =
        var_map.all_vars();

    assert!(
        !variables.is_empty(),
        "Model should contain trainable parameters"
    );

    let parameter_before: Vec<f32> =
        variables[0]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    let mut optimizer: AdamW =
        AdamW::new(
            variables.clone(),
            ParamsAdamW {
                lr: 0.001,
                ..Default::default()
            },
        )
        .unwrap();

    let logits: Tensor =
        model.forward(&input)
            .unwrap();

    let loss: Tensor =
        cross_entropy_loss(
            &logits,
            &targets,
        )
        .unwrap();

    optimizer
        .backward_step(&loss)
        .unwrap();

    let parameter_after: Vec<f32> =
        variables[0]
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

    assert_eq!(
        parameter_before.len(),
        parameter_after.len(),
        "Parameter size changed unexpectedly"
    );
}


#[test]
fn adamw_reduces_training_loss() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::cross_entropy_loss;

    let device: Device =
        Device::Cpu;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            20,
            4,
            8,
            2,
            16,
            1,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let input: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let targets: Tensor =
        Tensor::new(
            &[2u32, 3, 4, 5],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let variables =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            variables.clone(),
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )
        .unwrap();

    let logits_before: Tensor =
        model.forward(&input)
            .unwrap();

    let loss_before: Tensor =
        cross_entropy_loss(
            &logits_before,
            &targets,
        )
        .unwrap();

    let loss_before_value: f32 =
        loss_before
            .to_scalar::<f32>()
            .unwrap();

    optimizer
        .backward_step(&loss_before)
        .unwrap();

    let logits_after: Tensor =
        model.forward(&input)
            .unwrap();

    let loss_after: Tensor =
        cross_entropy_loss(
            &logits_after,
            &targets,
        )
        .unwrap();

    let loss_after_value: f32 =
        loss_after
            .to_scalar::<f32>()
            .unwrap();

    assert!(
        loss_after_value < loss_before_value,
        "Training loss did not decrease: before={}, after={}",
        loss_before_value,
        loss_after_value
    );
}

#[test]
fn orange_aide_can_overfit_tiny_dataset() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::cross_entropy_loss;

    let device: Device =
        Device::Cpu;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
            DType::F32,
            &device,
        );

    let model: OrangeAideModel =
        OrangeAideModel::new(
            20,
            4,
            8,
            2,
            16,
            1,
            1e-5,
            var_builder,
            &device,
        )
        .unwrap();

    let input: Tensor =
        Tensor::new(
            &[1u32, 2, 3, 4],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let targets: Tensor =
        Tensor::new(
            &[2u32, 3, 4, 5],
            &device,
        )
        .unwrap()
        .reshape((1, 4))
        .unwrap();

    let variables =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            variables,
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )
        .unwrap();

    let logits_before: Tensor =
        model.forward(&input)
            .unwrap();

    let initial_loss: Tensor =
        cross_entropy_loss(
            &logits_before,
            &targets,
        )
        .unwrap();

    let initial_loss_value: f32 =
        initial_loss
            .to_scalar::<f32>()
            .unwrap();

    for _step in 0..1_000 {
        let logits: Tensor =
            model.forward(&input)
                .unwrap();

        let loss: Tensor =
            cross_entropy_loss(
                &logits,
                &targets,
            )
            .unwrap();

        optimizer
            .backward_step(&loss)
            .unwrap();
    }

    let final_logits: Tensor =
        model.forward(&input)
            .unwrap();

    let final_loss: Tensor =
        cross_entropy_loss(
            &final_logits,
            &targets,
        )
        .unwrap();

    let final_loss_value: f32 =
        final_loss
            .to_scalar::<f32>()
            .unwrap();

    println!(
        "Tiny overfit loss: {} -> {}",
        initial_loss_value,
        final_loss_value
    );

    assert!(
        final_loss_value < initial_loss_value,
        "Training did not reduce the loss"
    );

    assert!(
        final_loss_value < 0.5,
        "orange_aide failed to overfit the tiny dataset: final loss = {}",
        final_loss_value
    );
}

#[test]
fn train_step_returns_loss_and_updates_parameters() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_step;

    let device: Device =
        Device::Cpu;

    let vocab_size: usize = 10;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 8;
    let num_heads: usize = 2;
    let hidden_dim: usize = 16;
    let num_layers: usize = 1;
    let eps: f64 = 1e-5;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
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
            eps,
            var_builder,
            &device,
        )?;

    let inputs: Tensor =
        Tensor::new(
            &[[1u32, 2, 3, 4]],
            &device,
        )?;

    let targets: Tensor =
        Tensor::new(
            &[[2u32, 3, 4, 5]],
            &device,
        )?;

    let params_before: Vec<f32> =
        var_map
            .all_vars()
            .iter()
            .flat_map(|tensor| {
                tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

    let params: Vec<candle_core::Var> =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            params,
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )?;

    let loss: f32 =
        train_step(
            &model,
            &mut optimizer,
            &inputs,
            &targets,
        )?;

    assert!(
        loss.is_finite(),
        "Loss must be finite"
    );

    assert!(
        loss > 0.0,
        "Loss should be positive"
    );

    let params_after: Vec<f32> =
        var_map
            .all_vars()
            .iter()
            .flat_map(|tensor| {
                tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

    assert_eq!(
        params_before.len(),
        params_after.len()
    );

    let changed: bool =
        params_before
            .iter()
            .zip(params_after.iter())
            .any(|(before, after)| {
                before != after
            });

    assert!(
        changed,
        "At least one parameter should change after train_step"
    );

    Ok(())
}

#[test]
fn train_model_reduces_loss() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_model;

    let device: Device =
        Device::Cpu;

    let vocab_size: usize = 10;
    let sequence_length: usize = 4;
    let embedding_dim: usize = 8;
    let num_heads: usize = 2;
    let hidden_dim: usize = 16;
    let num_layers: usize = 1;
    let eps: f64 = 1e-5;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
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
            eps,
            var_builder,
            &device,
        )?;

    let inputs: Tensor =
        Tensor::new(
            &[[1u32, 2, 3, 4]],
            &device,
        )?;

    let targets: Tensor =
        Tensor::new(
            &[[2u32, 3, 4, 5]],
            &device,
        )?;

    let params: Vec<candle_core::Var> =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            params,
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )?;

    let losses: Vec<f32> =
        train_model(
            &model,
            &mut optimizer,
            &inputs,
            &targets,
            100,
        )?;

    assert_eq!(
        losses.len(),
        100
    );

    assert!(
        losses[0].is_finite()
    );

    assert!(
        losses[99].is_finite()
    );

    assert!(
        losses[99] < losses[0],
        "Loss should decrease: {} -> {}",
        losses[0],
        losses[99]
    );

    Ok(())
}

#[test]
fn full_training_pipeline_works_with_tokenized_text() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::data::{
        batch_to_tensors,
        create_input_target_sequences,
        load_tokenizer,
        tokenize_text,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_step;

    let device: Device =
        Device::Cpu;

    let sequence_length: usize =
        8;

    let embedding_dim: usize =
        8;

    let num_heads: usize =
        2;

    let hidden_dim: usize =
        16;

    let num_layers: usize =
        1;

    let eps: f64 =
        1e-5;

    // Load the existing tokenizer.
    let tokenizer =
        load_tokenizer()?;

    let text: &str =
        "the cat sat on the mat the cat sat";

    // Text → token IDs.
    let token_ids: Vec<u32> =
        tokenize_text(
            &tokenizer,
            text,
        )?;

    assert!(
        !token_ids.is_empty(),
        "Tokenizer should produce token IDs"
    );

    // Token IDs → input/target sequences.
    let (
        input_sequences,
        target_sequences,
    ) =
        create_input_target_sequences(
            &token_ids,
            sequence_length,
        );

    assert!(
        !input_sequences.is_empty(),
        "Text should produce training sequences"
    );

    assert_eq!(
        input_sequences.len(),
        target_sequences.len()
    );

    assert_eq!(
        input_sequences[0].len(),
        target_sequences[0].len()
    );

    // Input/target sequences → tensors.
    let (
        inputs,
        targets,
    ): (Tensor, Tensor) =
        batch_to_tensors(
            &input_sequences,
            &target_sequences,
            &device,
        )?;

    assert_eq!(
        inputs.dims(),
        targets.dims()
    );

    assert_eq!(
        inputs.dims().len(),
        2
    );

    assert!(
        inputs.dims()[1] <= sequence_length
    );

    // Use the actual tokenizer vocabulary size.
    let vocab_size: usize =
        tokenizer.get_vocab_size(true);

    assert!(
        vocab_size > 0,
        "Tokenizer vocabulary must not be empty"
    );

    // Verify every token ID is valid.
    let input_values: Vec<u32> =
        inputs
            .flatten_all()?
            .to_vec1()?;

    let target_values: Vec<u32> =
        targets
            .flatten_all()?
            .to_vec1()?;

    for token_id in input_values
        .iter()
        .chain(target_values.iter())
    {
        assert!(
            (*token_id as usize) < vocab_size,
            "Token ID {} exceeds vocabulary size {}",
            token_id,
            vocab_size
        );
    }

    // Create model parameters.
    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
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
            eps,
            var_builder,
            &device,
        )?;

    // Create optimizer.
    let params: Vec<candle_core::Var> =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            params,
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )?;

    // Run one complete training step.
    let loss: f32 =
        train_step(
            &model,
            &mut optimizer,
            &inputs,
            &targets,
        )?;

    assert!(
        loss.is_finite(),
        "Training loss must be finite"
    );

    assert!(
        loss > 0.0,
        "Training loss should be positive"
    );

    Ok(())
}

#[test]
fn train_batches_processes_multiple_batches() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        AdamW,
        Optimizer,
        ParamsAdamW,
        VarBuilder,
        VarMap,
    };

    use orange_aide::model::OrangeAideModel;
    use orange_aide::train::train_batches;

    let device: Device =
        Device::Cpu;

    let vocab_size: usize =
        10;

    let sequence_length: usize =
        4;

    let embedding_dim: usize =
        8;

    let num_heads: usize =
        2;

    let hidden_dim: usize =
        16;

    let num_layers: usize =
        1;

    let eps: f64 =
        1e-5;

    let var_map: VarMap =
        VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(
            &var_map,
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
            eps,
            var_builder,
            &device,
        )?;

    let inputs_1: Tensor =
        Tensor::new(
            &[[1u32, 2, 3, 4]],
            &device,
        )?;

    let targets_1: Tensor =
        Tensor::new(
            &[[2u32, 3, 4, 5]],
            &device,
        )?;

    let inputs_2: Tensor =
        Tensor::new(
            &[[2u32, 3, 4, 5]],
            &device,
        )?;

    let targets_2: Tensor =
        Tensor::new(
            &[[3u32, 4, 5, 6]],
            &device,
        )?;

    let batches: Vec<(Tensor, Tensor)> =
        vec![
            (inputs_1, targets_1),
            (inputs_2, targets_2),
        ];

    let params: Vec<candle_core::Var> =
        var_map.all_vars();

    let mut optimizer: AdamW =
        AdamW::new(
            params,
            ParamsAdamW {
                lr: 0.01,
                ..Default::default()
            },
        )?;

    let losses: Vec<f32> =
        train_batches(
            &model,
            &mut optimizer,
            &batches,
        )?;

    assert_eq!(
        losses.len(),
        2
    );

    for loss in &losses {
        assert!(
            loss.is_finite(),
            "Every batch loss must be finite"
        );

        assert!(
            *loss > 0.0,
            "Every batch loss should be positive"
        );
    }

    Ok(())
}

#[test]
fn fixed_real_corpus_batch_loss_decreases() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{AdamW, VarBuilder, VarMap};
    use orange_aide::{data, model::OrangeAideModel, train};
    use std::path::Path;

    let device: Device = Device::Cpu;
    let sequence_length: usize = 16;
    let batch_size: usize = 2;

    // Load the actual project corpus and tokenizer.
    let corpus: String = data::load_file(Path::new("data/corpus.txt"))?;
    let tokenizer = data::load_tokenizer()?;
    let token_ids: Vec<u32> = data::tokenize_text(&tokenizer, &corpus)?;

    assert!(
        token_ids.len() >= sequence_length + batch_size,
        "Corpus does not contain enough tokens"
    );

    // Build a fixed batch. Both examples remain unchanged throughout training.
    let mut input_sequences: Vec<Vec<u32>> = Vec::new();
    let mut target_sequences: Vec<Vec<u32>> = Vec::new();

    for start in 0..batch_size {
        input_sequences.push(
            token_ids[start..start + sequence_length].to_vec()
        );
        target_sequences.push(
            token_ids[start + 1..start + sequence_length + 1].to_vec()
        );
    }

    let (inputs, targets): (Tensor, Tensor) =
        data::batch_to_tensors(&input_sequences, &target_sequences, &device)?;

    let vocab_size: usize = tokenizer.get_vocab_size(true);
    let embedding_dim: usize = 32;
    let num_heads: usize = 4;
    let hidden_dim: usize = 64;
    let num_layers: usize = 1;
    let learning_rate: f64 = 0.001;

    let mut varmap: VarMap = VarMap::new();
    let var_builder: VarBuilder =
        VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model: OrangeAideModel = OrangeAideModel::new(
        vocab_size,
        sequence_length,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers,
        1e-5,
        var_builder,
        &device,
    )?;

    let mut optimizer: AdamW =
        AdamW::new_lr(varmap.all_vars(), learning_rate)?;

    // Measure loss before any training updates.
    let initial_logits: Tensor = model.forward(&inputs)?;
    let initial_loss: f32 =
        train::cross_entropy_loss(&initial_logits, &targets)?
            .to_scalar::<f32>()?;

    // Train repeatedly on the exact same real-text batch.
    for _ in 0..100 {
        train::train_step(&model, &mut optimizer, &inputs, &targets)?;
    }

    // Evaluate the same batch again, without updating the model.
    let final_logits: Tensor = model.forward(&inputs)?;
    let final_loss: f32 =
        train::cross_entropy_loss(&final_logits, &targets)?
            .to_scalar::<f32>()?;

    println!(
        "Fixed real-corpus batch loss: {:.4} -> {:.4}",
        initial_loss,
        final_loss
    );

    assert!(initial_loss.is_finite() && final_loss.is_finite());
    assert!(
        final_loss < initial_loss,
        "Loss did not decrease on the fixed real-corpus batch"
    );

    Ok(())
}

#[test]
fn evaluate_loss_returns_finite_value() -> anyhow::Result<()> {
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use orange_aide::{data, model::OrangeAideModel, train};
    use std::path::Path;

    let device: Device = Device::Cpu;
    let sequence_length: usize = 16;
    let batch_size: usize = 2;

    let corpus: String = data::load_file(Path::new("data/corpus.txt"))?;
    let tokenizer = data::load_tokenizer()?;
    let token_ids: Vec<u32> = data::tokenize_text(&tokenizer, &corpus)?;

    let (_, validation_tokens) =
        data::split_tokens(&token_ids, 0.2)?;

    let vocab_size: usize = tokenizer.get_vocab_size(true);
    let mut varmap: VarMap = VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model: OrangeAideModel = OrangeAideModel::new(
        vocab_size,
        sequence_length,
        32,
        4,
        64,
        1,
        1e-5,
        var_builder,
        &device,
    )?;

    let loss: f32 = train::evaluate_loss(
        &model,
        &validation_tokens,
        sequence_length,
        batch_size,
        &device,
    )?;

    assert!(loss.is_finite());
    assert!(loss > 0.0);

    Ok(())
}

#[test]
fn evaluate_loss_rejects_invalid_inputs() -> anyhow::Result<()> {
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use orange_aide::model::OrangeAideModel;

    let device: Device = Device::Cpu;
    let mut varmap: VarMap = VarMap::new();

    let var_builder: VarBuilder =
        VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    let model: OrangeAideModel = OrangeAideModel::new(
        100,
        4,
        8,
        2,
        16,
        1,
        1e-5,
        var_builder,
        &device,
    )?;

    assert!(
        orange_aide::train::evaluate_loss(
            &model, &[], 4, 1, &device
        ).is_err()
    );

    assert!(
        orange_aide::train::evaluate_loss(
            &model, &[1, 2, 3, 4], 0, 1, &device
        ).is_err()
    );

    assert!(
        orange_aide::train::evaluate_loss(
            &model, &[1, 2, 3, 4], 4, 1, &device
        ).is_err()
    );

    assert!(
        orange_aide::train::evaluate_loss(
            &model, &[1, 2, 3, 4, 5], 4, 0, &device
        ).is_err()
    );

    Ok(())
}
#[test]
fn validation_window_starts_do_not_overlap() {
    use orange_aide::train;

    let starts: Vec<usize> = train::validation_window_starts(100, 16);

    assert_eq!(starts, vec![0, 16, 32, 48, 64, 80]);
}

#[test]
fn validation_window_starts_include_exact_fit() {
    use orange_aide::train;

    // 17 tokens = 16 inputs + 1 extra token for the final target.
    let starts: Vec<usize> = train::validation_window_starts(17, 16);

    assert_eq!(starts, vec![0]);
}

#[test]
fn validation_window_starts_are_empty_without_room_for_a_target() {
    use orange_aide::train;

    let starts: Vec<usize> = train::validation_window_starts(16, 16);

    assert!(starts.is_empty());
}

#[test]
fn validation_window_starts_never_run_past_the_tokens() {
    use orange_aide::train;

    let token_count: usize = 1_003;
    let sequence_length: usize = 64;

    for start in train::validation_window_starts(token_count, sequence_length) {
        assert!(start + sequence_length + 1 <= token_count);
    }
}

#[test]
fn is_interval_step_fires_on_multiples() {
    use orange_aide::train::is_interval_step;

    assert!(!is_interval_step(4, 5));
    assert!(is_interval_step(5, 5));
    assert!(!is_interval_step(6, 5));
    assert!(is_interval_step(10, 5));
}

#[test]
fn is_interval_step_never_fires_when_interval_is_zero() {
    use orange_aide::train::is_interval_step;

    assert!(!is_interval_step(5, 0));
}

#[test]
fn best_validation_first_loss_is_best() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::new();

    assert!(tracker.update(5.0));
    assert_eq!(tracker.best(), Some(5.0));
    assert_eq!(tracker.evaluations_since_best(), 0);
}

#[test]
fn best_validation_only_improves_on_strictly_lower_loss() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::new();

    assert!(tracker.update(5.0));
    assert!(!tracker.update(6.0));
    assert_eq!(tracker.evaluations_since_best(), 1);
    assert!(tracker.update(4.0));
    assert_eq!(tracker.evaluations_since_best(), 0);
    assert!(!tracker.update(4.0));
    assert_eq!(tracker.best(), Some(4.0));
}

#[test]
fn best_validation_ignores_non_finite_losses() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::new();

    assert!(!tracker.update(f32::NAN));
    assert_eq!(tracker.best(), None);

    assert!(tracker.update(3.0));
    assert!(!tracker.update(f32::INFINITY));
    assert_eq!(tracker.best(), Some(3.0));
}

#[test]
fn best_validation_should_stop_respects_patience() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::new();

    tracker.update(5.0);
    tracker.update(6.0);
    tracker.update(6.0);
    assert!(!tracker.should_stop(3));

    tracker.update(6.0);
    assert!(tracker.should_stop(3));

    // A patience of zero disables early stopping.
    assert!(!tracker.should_stop(0));
}

#[test]
fn best_device_returns_a_usable_device() {
    let device = orange_aide::train::best_device().unwrap();

    let tensor = candle_core::Tensor::new(&[1f32, 2.0, 3.0], &device).unwrap();
    let sum: f32 = tensor.sum_all().unwrap().to_scalar::<f32>().unwrap();

    assert_eq!(sum, 6.0);
}

fn tiny_model_and_varmap(
    device: &candle_core::Device,
) -> (candle_nn::VarMap, orange_aide::model::OrangeAideModel) {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    let varmap: VarMap = VarMap::new();
    let var_builder: VarBuilder =
        VarBuilder::from_varmap(&varmap, DType::F32, device);

    let model = orange_aide::model::OrangeAideModel::new(
        50, 8, 16, 2, 32, 2, 1e-5, var_builder, device,
    )
    .unwrap();

    (varmap, model)
}

fn fresh_temp_dir(name: &str) -> std::path::PathBuf {
    let dir: std::path::PathBuf = std::env::temp_dir().join(format!(
        "orange_aide_{}_{}",
        name,
        std::process::id()
    ));

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    dir
}

#[test]
fn train_corpus_monitored_saves_best_and_latest_checkpoints() {
    use candle_nn::{AdamW, Optimizer};
    use orange_aide::train::{self, MonitorConfig};

    let device = candle_core::Device::Cpu;
    let (varmap, model) = tiny_model_and_varmap(&device);
    let mut optimizer: AdamW = AdamW::new_lr(varmap.all_vars(), 1e-2).unwrap();

    let training_tokens: Vec<u32> = (0..400u32).map(|i| i % 10).collect();
    let validation_tokens: Vec<u32> = (0..100u32).map(|i| i % 10).collect();

    let dir = fresh_temp_dir("monitored_saves");
    let best = dir.join("best.safetensors");
    let latest = dir.join("latest.safetensors");

    let monitor = MonitorConfig {
        eval_interval: 5,
        checkpoint_interval: 10,
        patience: 0,
        best_checkpoint: &best,
        latest_checkpoint: &latest,
    };

    let losses: Vec<f32> = train::train_corpus_monitored(
        &model,
        &mut optimizer,
        &varmap,
        &training_tokens,
        &validation_tokens,
        8,
        4,
        20,
        &monitor,
        &device,
    )
    .unwrap();

    assert_eq!(losses.len(), 20);
    assert!(best.is_file());
    assert!(latest.is_file());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn train_corpus_monitored_stops_early_when_validation_stops_improving() {
    use candle_nn::{AdamW, Optimizer};
    use orange_aide::train::{self, MonitorConfig};

    let device = candle_core::Device::Cpu;
    let (varmap, model) = tiny_model_and_varmap(&device);

    // A learning rate of zero keeps the weights fixed, so validation loss
    // can never improve after the first evaluation.
    let mut optimizer: AdamW = AdamW::new_lr(varmap.all_vars(), 0.0).unwrap();

    let training_tokens: Vec<u32> = (0..400u32).map(|i| i % 10).collect();
    let validation_tokens: Vec<u32> = (0..100u32).map(|i| i % 10).collect();

    let dir = fresh_temp_dir("monitored_early_stop");
    let best = dir.join("best.safetensors");
    let latest = dir.join("latest.safetensors");

    let monitor = MonitorConfig {
        eval_interval: 1,
        checkpoint_interval: 0,
        patience: 2,
        best_checkpoint: &best,
        latest_checkpoint: &latest,
    };

    let losses: Vec<f32> = train::train_corpus_monitored(
        &model,
        &mut optimizer,
        &varmap,
        &training_tokens,
        &validation_tokens,
        8,
        4,
        50,
        &monitor,
        &device,
    )
    .unwrap();

    assert!(losses.len() < 50);
        // Nothing beat the starting validation loss, so no best checkpoint exists.
    assert!(!best.is_file());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn train_corpus_monitored_rejects_zero_eval_interval() {
    use candle_nn::{AdamW, Optimizer};
    use orange_aide::train::{self, MonitorConfig};

    let device = candle_core::Device::Cpu;
    let (varmap, model) = tiny_model_and_varmap(&device);
    let mut optimizer: AdamW = AdamW::new_lr(varmap.all_vars(), 1e-2).unwrap();

    let tokens: Vec<u32> = (0..400u32).map(|i| i % 10).collect();

    let dir = fresh_temp_dir("monitored_zero_interval");
    let best = dir.join("best.safetensors");
    let latest = dir.join("latest.safetensors");

    let monitor = MonitorConfig {
        eval_interval: 0,
        checkpoint_interval: 0,
        patience: 0,
        best_checkpoint: &best,
        latest_checkpoint: &latest,
    };

    let result = train::train_corpus_monitored(
        &model, &mut optimizer, &varmap, &tokens, &tokens, 8, 4, 10, &monitor, &device,
    );

    assert!(result.is_err());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn best_validation_with_baseline_requires_strict_improvement() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::with_baseline(5.0);

    assert_eq!(tracker.best(), Some(5.0));
    assert!(!tracker.update(5.0));
    assert!(!tracker.update(6.0));
    assert!(tracker.update(4.5));
    assert_eq!(tracker.best(), Some(4.5));
}

#[test]
fn best_validation_with_non_finite_baseline_starts_empty() {
    use orange_aide::train::BestValidation;

    let mut tracker: BestValidation = BestValidation::with_baseline(f32::NAN);

    assert_eq!(tracker.best(), None);
    assert!(tracker.update(9.0));
}

#[test]
fn constant_learning_rate_schedule_never_changes() {
    use orange_aide::train::LearningRateSchedule;

    let schedule = LearningRateSchedule::constant(0.003);

    for step in [1usize, 10, 500, 1000] {
        assert!((schedule.at_step(step, 1000) - 0.003).abs() < 1e-12);
    }
}

#[test]
fn learning_rate_schedule_warms_up_linearly() {
    use orange_aide::train::LearningRateSchedule;

    let schedule = LearningRateSchedule {
        base_lr: 1.0,
        warmup_steps: 4,
        min_lr_fraction: 0.1,
    };

    assert!((schedule.at_step(1, 100) - 0.25).abs() < 1e-12);
    assert!((schedule.at_step(2, 100) - 0.5).abs() < 1e-12);
    assert!((schedule.at_step(4, 100) - 1.0).abs() < 1e-12);
}

#[test]
fn learning_rate_schedule_decays_to_the_minimum_fraction() {
    use orange_aide::train::LearningRateSchedule;

    let schedule = LearningRateSchedule {
        base_lr: 1.0,
        warmup_steps: 2,
        min_lr_fraction: 0.1,
    };

    let mut previous: f64 = schedule.at_step(2, 12);
    assert!((previous - 1.0).abs() < 1e-12);

    for step in 3..=12 {
        let current: f64 = schedule.at_step(step, 12);
        assert!(current < previous, "learning rate did not decrease at step {step}");
        previous = current;
    }

    assert!((schedule.at_step(12, 12) - 0.1).abs() < 1e-9);
}

#[test]
fn learning_rate_schedule_midpoint_is_halfway() {
    use orange_aide::train::LearningRateSchedule;

    let schedule = LearningRateSchedule {
        base_lr: 1.0,
        warmup_steps: 0,
        min_lr_fraction: 0.0,
    };

    assert!((schedule.at_step(50, 100) - 0.5).abs() < 1e-9);
}

#[test]
fn rng_state_from_seed_differs_per_seed_and_is_never_zero() {
    use orange_aide::train::rng_state_from_seed;

    for seed in 0..100u64 {
        assert_ne!(rng_state_from_seed(seed), 0);
    }

    assert_ne!(rng_state_from_seed(1), rng_state_from_seed(2));
    assert_eq!(rng_state_from_seed(7), rng_state_from_seed(7));
}

#[test]
fn sample_window_starts_are_deterministic_and_in_range() {
    use orange_aide::train::{rng_state_from_seed, sample_window_starts};

    let mut rng_a: u64 = rng_state_from_seed(11);
    let mut rng_b: u64 = rng_state_from_seed(11);
    let mut rng_c: u64 = rng_state_from_seed(12);

    let a: Vec<usize> = sample_window_starts(&mut rng_a, 1000, 64);
    let b: Vec<usize> = sample_window_starts(&mut rng_b, 1000, 64);
    let c: Vec<usize> = sample_window_starts(&mut rng_c, 1000, 64);

    assert_eq!(a.len(), 64);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(a.iter().all(|&start| start < 1000));
}

#[test]
fn train_corpus_with_schedule_ends_at_the_minimum_learning_rate() {
    use candle_nn::{AdamW, Optimizer};
    use orange_aide::train::{self, LearningRateSchedule, MonitorConfig};

    let device = candle_core::Device::Cpu;
    let (varmap, model) = tiny_model_and_varmap(&device);
    let mut optimizer: AdamW = AdamW::new_lr(varmap.all_vars(), 0.5).unwrap();

    let training_tokens: Vec<u32> = (0..400u32).map(|i| i % 10).collect();
    let validation_tokens: Vec<u32> = (0..100u32).map(|i| i % 10).collect();

    let dir = fresh_temp_dir("scheduled_lr");
    let best = dir.join("best.safetensors");
    let latest = dir.join("latest.safetensors");

    let monitor = MonitorConfig {
        eval_interval: 5,
        checkpoint_interval: 0,
        patience: 0,
        best_checkpoint: &best,
        latest_checkpoint: &latest,
    };

    let schedule = LearningRateSchedule {
        base_lr: 1e-2,
        warmup_steps: 2,
        min_lr_fraction: 0.1,
    };

    let losses: Vec<f32> = train::train_corpus_with_schedule(
        &model,
        &mut optimizer,
        &varmap,
        &training_tokens,
        &validation_tokens,
        8,
        4,
        10,
        &monitor,
        Some(schedule),
        5,
        &device,
    )
    .unwrap();

    assert_eq!(losses.len(), 10);
    assert!((optimizer.learning_rate() - 1e-3).abs() < 1e-9);

    std::fs::remove_dir_all(&dir).unwrap();
}
fn tiny_model_with_layers(
    device: &candle_core::Device,
    layers: usize,
) -> (candle_nn::VarMap, orange_aide::model::OrangeAideModel) {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    let varmap: VarMap = VarMap::new();
    let var_builder: VarBuilder =
        VarBuilder::from_varmap(&varmap, DType::F32, device);

    let model = orange_aide::model::OrangeAideModel::new(
        50, 8, 16, 2, 32, layers, 1e-5, var_builder, device,
    )
    .unwrap();

    (varmap, model)
}

fn std_of(tensor: &candle_core::Tensor) -> f32 {
    let values: Vec<f32> = tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;

    (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32)
        .sqrt()
}

#[test]
fn gpt_init_sets_small_weights_zero_biases_and_keeps_norms() {
    use orange_aide::train::apply_gpt_init;

    let device = candle_core::Device::Cpu;
    let (varmap, _model) = tiny_model_with_layers(&device, 2);

    apply_gpt_init(&varmap, 2, &device).unwrap();

    let data = varmap.data().lock().unwrap();

    for (name, var) in data.iter() {
        let tensor = var.as_tensor();

        if name.contains("norm") {
            let expected: f32 = if name.ends_with(".weight") { 1.0 } else { 0.0 };
            let values: Vec<f32> = tensor.to_vec1::<f32>().unwrap();
            assert!(
                values.iter().all(|v| (v - expected).abs() < 1e-6),
                "{name} should be unchanged"
            );
        } else if name.ends_with(".bias") {
            let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(values.iter().all(|v| *v == 0.0), "{name} should be zero");
        } else if name.contains("attention.output") || name.contains("feed_forward.second") {
            // 0.02 / sqrt(2 * 2 layers) = 0.01
            let s: f32 = std_of(tensor);
            assert!(s > 0.005 && s < 0.015, "{name} std {s} not near 0.01");
        } else {
            let s: f32 = std_of(tensor);
            assert!(s > 0.012 && s < 0.03, "{name} std {s} not near 0.02");
        }
    }
}

#[test]
fn gpt_init_shrinks_residual_projections_as_depth_grows() {
    use orange_aide::train::apply_gpt_init;

    let device = candle_core::Device::Cpu;

    let (varmap_shallow, _m1) = tiny_model_with_layers(&device, 2);
    let (varmap_deep, _m2) = tiny_model_with_layers(&device, 8);

    apply_gpt_init(&varmap_shallow, 2, &device).unwrap();
    apply_gpt_init(&varmap_deep, 8, &device).unwrap();

    let name: &str = "block_0.attention.output.weight";

    let shallow: f32 = std_of(
        varmap_shallow.data().lock().unwrap().get(name).unwrap().as_tensor(),
    );
    let deep: f32 = std_of(
        varmap_deep.data().lock().unwrap().get(name).unwrap().as_tensor(),
    );

    assert!(shallow > 1.5 * deep, "shallow {shallow} vs deep {deep}");
}

#[test]
fn gpt_init_starts_near_random_guess_loss() {
    use candle_core::Tensor;
    use orange_aide::train::{apply_gpt_init, cross_entropy_loss};

    let device = candle_core::Device::Cpu;
    let (varmap, model) = tiny_model_with_layers(&device, 2);

    apply_gpt_init(&varmap, 2, &device).unwrap();

    let inputs: Tensor =
        Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8], [8u32, 7, 6, 5, 4, 3, 2, 1]], &device)
            .unwrap();
    let targets: Tensor =
        Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9], [7u32, 6, 5, 4, 3, 2, 1, 0]], &device)
            .unwrap();

    let logits: Tensor = model.forward(&inputs).unwrap();
    let loss: f32 = cross_entropy_loss(&logits, &targets)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let random_guess: f32 = (50f32).ln();

    assert!(
        (loss - random_guess).abs() < 0.25,
        "starting loss {loss} is not near ln(50) = {random_guess}"
    );
}

#[test]
fn gpt_init_rejects_zero_layers() {
    use orange_aide::train::apply_gpt_init;

    let device = candle_core::Device::Cpu;
    let (varmap, _model) = tiny_model_with_layers(&device, 2);

    assert!(apply_gpt_init(&varmap, 0, &device).is_err());
}









