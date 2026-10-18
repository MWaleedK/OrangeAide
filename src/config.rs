#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub sequence_length: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,

    pub learning_rate: f64,
    pub training_steps: usize,
    pub batch_size: usize,

    pub layer_norm_eps: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            sequence_length: 64,
            embedding_dim: 128,
            num_heads: 4,
            hidden_dim: 512,
            num_layers: 4,

            learning_rate: 0.001,
            training_steps: 50000,
            batch_size: 64,

            layer_norm_eps: 1e-5,
        }
    }
}
