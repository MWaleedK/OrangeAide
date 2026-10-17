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
            embedding_dim: 256,
            num_heads: 8,
            hidden_dim: 1024,
            num_layers: 6,

            learning_rate: 0.003,
            training_steps: 10000,
            batch_size: 32,

            layer_norm_eps: 1e-5,
        }
    }
}
