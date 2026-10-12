use orange_aide::data::create_input_target_sequences;
use std::path::Path;
#[test]
fn input_target_are_shifted_by_one() {
    let token_ids: Vec<u32> = vec![10, 20, 30, 40, 50];

    let (inputs, targets): (Vec<Vec<u32>>, Vec<Vec<u32>>) =
        create_input_target_sequences(&token_ids, 3);

    assert_eq!(inputs[0], vec![10, 20, 30]);
    assert_eq!(targets[0], vec![20, 30, 40]);

    assert_eq!(inputs[1], vec![20, 30, 40]);
    assert_eq!(targets[1], vec![30, 40, 50]);
}

#[test]
fn batch_has_correct_size() {
    let token_ids: Vec<u32> =
        vec![10, 20, 30, 40, 50, 60];

    let result =
        orange_aide::data::create_batch(
            &token_ids,
            3,
            2,
            0,
        );

    let (inputs, targets) =
        result.expect("Batch should have been created");

    assert_eq!(inputs.len(), 2);
    assert_eq!(targets.len(), 2);
}

#[test]
fn batch_contains_shifted_sequences() {
    let token_ids: Vec<u32> =
        vec![10, 20, 30, 40, 50, 60];

    let result =
        orange_aide::data::create_batch(
            &token_ids,
            3,
            2,
            0,
        );

    let (inputs, targets) =
        result.expect("Batch should have been created");

    assert_eq!(inputs[0], vec![10, 20, 30]);
    assert_eq!(targets[0], vec![20, 30, 40]);

    assert_eq!(inputs[1], vec![20, 30, 40]);
    assert_eq!(targets[1], vec![30, 40, 50]);
}

#[test]
fn batch_can_start_at_different_position() {
    let token_ids: Vec<u32> =
        vec![10, 20, 30, 40, 50, 60];

    let result =
        orange_aide::data::create_batch(
            &token_ids,
            3,
            2,
            1,
        );

    let (inputs, targets) =
        result.expect("Batch should have been created");

    assert_eq!(inputs[0], vec![20, 30, 40]);
    assert_eq!(targets[0], vec![30, 40, 50]);

    assert_eq!(inputs[1], vec![30, 40, 50]);
    assert_eq!(targets[1], vec![40, 50, 60]);
}

#[test]
fn batch_to_tensors_has_correct_shapes() {
    use candle_core::Device;

    let inputs: Vec<Vec<u32>> =
        vec![
            vec![10, 20, 30],
            vec![20, 30, 40],
        ];

    let targets: Vec<Vec<u32>> =
        vec![
            vec![20, 30, 40],
            vec![30, 40, 50],
        ];

    let device: Device = Device::Cpu;

    let (input_tensor, target_tensor) =
        orange_aide::data::batch_to_tensors(
            &inputs,
            &targets,
            &device,
        )
        .expect("Failed to create tensors");

    assert_eq!(input_tensor.dims(), &[2, 3]);
    assert_eq!(target_tensor.dims(), &[2, 3]);
}

#[test]
fn batch_to_tensors_preserves_values() {
    use candle_core::Device;

    let inputs: Vec<Vec<u32>> =
        vec![
            vec![10, 20, 30],
            vec![20, 30, 40],
        ];

    let targets: Vec<Vec<u32>> =
        vec![
            vec![20, 30, 40],
            vec![30, 40, 50],
        ];

    let device: Device = Device::Cpu;

    let (input_tensor, target_tensor) =
        orange_aide::data::batch_to_tensors(
            &inputs,
            &targets,
            &device,
        )
        .expect("Failed to create tensors");

    let input_values: Vec<Vec<u32>> =
        input_tensor.to_vec2().expect("Failed to read input tensor");

    let target_values: Vec<Vec<u32>> =
        target_tensor.to_vec2().expect("Failed to read target tensor");

    assert_eq!(input_values, inputs);
    assert_eq!(target_values, targets);
}


#[test]
fn batch_returns_none_when_not_enough_tokens() {
    let token_ids: Vec<u32> =
        vec![10, 20, 30, 40];

    let result =
        orange_aide::data::create_batch(
            &token_ids,
            3,
            2,
            0,
        );

    assert!(result.is_none());
}

#[test]
fn input_and_target_have_same_length() {
    let token_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

    let (inputs, targets): (Vec<Vec<u32>>, Vec<Vec<u32>>) =
        create_input_target_sequences(&token_ids, 4);

    assert_eq!(inputs.len(), targets.len());

    for i in 0..inputs.len() {
        assert_eq!(inputs[i].len(), targets[i].len());
    }
}

#[test]
fn text_file_can_be_loaded() {
    let path: &Path = Path::new("data/corpus.txt");

    let text: String = orange_aide::data::load_file(path)
        .expect("Failed to load text file");

    assert!(!text.trim().is_empty());
}

#[test]
fn sequence_too_short_produces_no_sequences() {
    let token_ids: Vec<u32> = vec![1, 2, 3];

    let (inputs, targets): (Vec<Vec<u32>>, Vec<Vec<u32>>) =
        create_input_target_sequences(&token_ids, 3);

    assert!(inputs.is_empty());
    assert!(targets.is_empty());
}

#[test]
fn batch_tensors_are_valid_for_training() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};

    use orange_aide::data::batch_to_tensors;

    let device: Device =
        Device::Cpu;

    let input_sequences: Vec<Vec<u32>> =
        vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        ];

    let target_sequences: Vec<Vec<u32>> =
        vec![
            vec![2, 3, 4, 5],
            vec![6, 7, 8, 9],
        ];

    let (inputs, targets): (Tensor, Tensor) =
        batch_to_tensors(
            &input_sequences,
            &target_sequences,
            &device,
        )?;

    assert_eq!(
        inputs.dims(),
        &[2, 4]
    );

    assert_eq!(
        targets.dims(),
        &[2, 4]
    );

    assert_eq!(
        inputs.to_vec2::<u32>()?,
        vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        ]
    );

    assert_eq!(
        targets.to_vec2::<u32>()?,
        vec![
            vec![2, 3, 4, 5],
            vec![6, 7, 8, 9],
        ]
    );

    Ok(())
}


#[test]
fn create_batch_returns_correct_input_target_pairs()
    -> anyhow::Result<()>{
    use orange_aide::data::create_batch;

    let token_ids: Vec<u32> =
        vec![
            10, 11, 12, 13, 14, 15, 16, 17
        ];

    let sequence_length: usize =
        3;

    let batch_size: usize =
        2;

    let start_index: usize =
        0;

    let batch =
        create_batch(
            &token_ids,
            sequence_length,
            batch_size,
            start_index,
        );

    assert!(
        batch.is_some(),
        "Expected a batch to be created"
    );

    let (
        inputs,
        targets,
    ) =
        batch.unwrap();

    assert_eq!(
        inputs.len(),
        2
    );

    assert_eq!(
        targets.len(),
        2
    );

    assert_eq!(
        inputs[0],
        vec![10, 11, 12]
    );

    assert_eq!(
        targets[0],
        vec![11, 12, 13]
    );

    assert_eq!(
        inputs[1],
        vec![11, 12, 13]
    );

    assert_eq!(
        targets[1],
        vec![12, 13, 14]
    );

    Ok(())
}

#[test]
fn create_batch_returns_none_when_not_enough_tokens()
    -> anyhow::Result<()>
{
    use orange_aide::data::create_batch;

    let token_ids: Vec<u32> =
        vec![1, 2, 3, 4];

    let result =
        create_batch(
            &token_ids,
            3,
            2,
            0,
        );

    assert!(
        result.is_none(),
        "Batch should not be created when insufficient tokens exist"
    );

    Ok(())
}

#[test]
fn split_tokens_preserves_order_and_separates_validation() {
    use orange_aide::data::split_tokens;

    let token_ids: Vec<u32> = (0..100).collect();

    let (training_tokens, validation_tokens) =
        split_tokens(&token_ids, 0.2).expect("Split should succeed");

    assert_eq!(training_tokens.len(), 80);
    assert_eq!(validation_tokens.len(), 20);

    assert_eq!(training_tokens, (0..80).collect::<Vec<u32>>());
    assert_eq!(validation_tokens, (80..100).collect::<Vec<u32>>());

    assert!(
        training_tokens
            .iter()
            .all(|token| !validation_tokens.contains(token)),
        "Training and validation tokens must not overlap"
    );
}

#[test]
fn split_tokens_rejects_invalid_validation_fraction() {
    use orange_aide::data::split_tokens;

    let token_ids: Vec<u32> = (0..10).collect();

    assert!(split_tokens(&token_ids, 0.0).is_err());
    assert!(split_tokens(&token_ids, 1.0).is_err());
    assert!(split_tokens(&token_ids, -0.1).is_err());
    assert!(split_tokens(&token_ids, 1.1).is_err());
    assert!(split_tokens(&token_ids, f64::NAN).is_err());
}

#[test]
fn split_tokens_rejects_empty_or_too_small_input() {
    use orange_aide::data::split_tokens;

    let empty: Vec<u32> = Vec::new();
    assert!(split_tokens(&empty, 0.2).is_err());

    let short: Vec<u32> = vec![42];
    assert!(split_tokens(&short, 0.2).is_err());
}
fn make_temp_dir(name: &str) -> std::path::PathBuf {
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
fn list_text_files_returns_only_txt_files_sorted_by_name() {
    let dir = make_temp_dir("list_text_files");

    std::fs::write(dir.join("b.txt"), "second").unwrap();
    std::fs::write(dir.join("a.txt"), "first").unwrap();
    std::fs::write(dir.join("notes.md"), "ignored").unwrap();
    std::fs::write(dir.join("C.TXT"), "uppercase extension").unwrap();

    let files = orange_aide::data::list_text_files(&dir).unwrap();

    let names: Vec<String> = files
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    assert_eq!(names, vec!["C.TXT", "a.txt", "b.txt"]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn load_directory_joins_all_txt_files_in_sorted_order() {
    let dir = make_temp_dir("load_directory");

    std::fs::write(dir.join("b.txt"), "second book").unwrap();
    std::fs::write(dir.join("a.txt"), "first book").unwrap();
    std::fs::write(dir.join("ignored.csv"), "not text").unwrap();

    let text: String = orange_aide::data::load_directory(&dir).unwrap();

    assert_eq!(text, "first book\n\nsecond book");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn load_directory_skips_empty_files() {
    let dir = make_temp_dir("load_directory_empty_file");

    std::fs::write(dir.join("a.txt"), "real text").unwrap();
    std::fs::write(dir.join("b.txt"), "   \n  ").unwrap();

    let text: String = orange_aide::data::load_directory(&dir).unwrap();

    assert_eq!(text, "real text");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn load_directory_rejects_directory_without_text_files() {
    let dir = make_temp_dir("load_directory_no_text");

    std::fs::write(dir.join("data.csv"), "a,b").unwrap();

    assert!(orange_aide::data::load_directory(&dir).is_err());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn load_directory_rejects_path_that_is_not_a_directory() {
    let missing: std::path::PathBuf =
        std::env::temp_dir().join("orange_aide_definitely_missing_dir");

    assert!(orange_aide::data::load_directory(&missing).is_err());
}

































