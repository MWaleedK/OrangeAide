use std::path::Path;
use tokenizers::Tokenizer;

#[test]
fn tokenizer_file_exists() {
    assert!(
        Path::new("tokenizer/tokenizer.json").exists(),
        "tokenizer.json was not created"
    );
}

#[test]
fn tokenizer_can_load() {
    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json")
        .expect("Failed to load tokenizer");

    assert!(tokenizer.get_vocab_size(false) > 0);
}

#[test]
fn tokenizer_can_encode() {
    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json")
        .expect("Failed to load tokenizer");

    let encoding = tokenizer
        .encode("The Verdict", true)
        .expect("Failed to encode text");

    assert!(!encoding.get_ids().is_empty());
}

#[test]
fn tokenizer_produces_valid_ids() {
    let tokenizer = Tokenizer::from_file("tokenizer/tokenizer.json")
        .expect("Failed to load tokenizer");

    let encoding = tokenizer
        .encode("The Verdict", true)
        .expect("Failed to encode text");

    let vocab_size = tokenizer.get_vocab_size(false);

    for id in encoding.get_ids() {
        assert!(
            (*id as usize) < vocab_size,
            "Token ID {} is outside vocabulary size {}",
            id,
            vocab_size
        );
    }
}