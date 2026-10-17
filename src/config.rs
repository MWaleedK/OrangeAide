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
            sequence_length: 128,
            embedding_dim: 256,
            num_heads: 32,
            hidden_dim: 2048,
            num_layers: 32,

            learning_rate: 0.005,
            training_steps: 20000,
            batch_size: 128,

            layer_norm_eps: 1e-5,
        }
    }
}
