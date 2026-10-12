use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear};

pub struct QkvProjection {
    query: Linear,
    key: Linear,
    value: Linear,
}

impl QkvProjection {
    pub fn new(embedding_dim: usize, var_builder: VarBuilder) -> Result<Self> {
        let query: Linear = linear(embedding_dim, embedding_dim, var_builder.pp("query"))?;

        let key: Linear = linear(embedding_dim, embedding_dim, var_builder.pp("key"))?;

        let value: Linear = linear(embedding_dim, embedding_dim, var_builder.pp("value"))?;

        Ok(Self { query, key, value })
    }

    pub fn forward(&self, input: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        let query: Tensor = self.query.forward(input)?;

        let key: Tensor = self.key.forward(input)?;

        let value: Tensor = self.value.forward(input)?;

        Ok((query, key, value))
    }
}

pub fn scaled_attention_scores(query: &Tensor, key: &Tensor) -> Result<Tensor> {
    if query.dims().len() != 3 {
        anyhow::bail!(
            "Query must have shape [batch, sequence, embedding], got {:?}",
            query.dims()
        );
    }

    if key.dims().len() != 3 {
        anyhow::bail!(
            "Key must have shape [batch, sequence, embedding], got {:?}",
            key.dims()
        );
    }

    if query.dims()[0] != key.dims()[0] {
        anyhow::bail!("Query and key batch sizes do not match");
    }

    if query.dims()[2] != key.dims()[2] {
        anyhow::bail!("Query and key embedding dimensions do not match");
    }

    let embedding_dim: usize = query.dims()[2];

    let key_transposed: Tensor = key.transpose(1, 2)?;

    let scores: Tensor = query.matmul(&key_transposed)?;

    let scale: f64 = (embedding_dim as f64).sqrt();

    let scores: Tensor = scores.affine(1.0 / scale, 0.0)?;

    Ok(scores)
}

pub struct PositionalEmbedding {
    embedding: Embedding,
    sequence_length: usize,
}

impl PositionalEmbedding {
    pub fn new(sequence_length: usize, embedding_dim: usize, device: &Device) -> Result<Self> {
        let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, device);

        let embedding: Embedding = embedding(sequence_length, embedding_dim, var_builder)?;

        Ok(Self {
            embedding,
            sequence_length,
        })
    }

    pub fn new_with_var_builder(
        sequence_length: usize,
        embedding_dim: usize,
        var_builder: VarBuilder,
    ) -> Result<Self> {
        let embedding: Embedding = embedding(sequence_length, embedding_dim, var_builder)?;

        Ok(Self {
            embedding,
            sequence_length,
        })
    }

    pub fn forward(&self, token_embeddings: &Tensor) -> Result<Tensor> {
        let dimensions: &[usize] = token_embeddings.dims();

        if dimensions.len() != 3 {
            anyhow::bail!(
                "Expected token embeddings with 3 dimensions [batch, sequence, embedding], got {:?}",
                dimensions
            );
        }

        let batch_size: usize = dimensions[0];
        let sequence_length: usize = dimensions[1];

        if sequence_length > self.sequence_length {
            anyhow::bail!(
                "Sequence length {} exceeds maximum positional embedding length {}",
                sequence_length,
                self.sequence_length
            );
        }

        let positions: Vec<u32> = (0..sequence_length)
            .map(|position| position as u32)
            .collect();

        let position_ids: Tensor = Tensor::new(positions.as_slice(), token_embeddings.device())?;

        let position_embeddings: Tensor = self.embedding.forward(&position_ids)?;

        let position_embeddings: Tensor = position_embeddings.unsqueeze(0)?.broadcast_as((
            batch_size,
            sequence_length,
            position_embeddings.dim(1)?,
        ))?;

        let output: Tensor = token_embeddings.add(&position_embeddings)?;

        Ok(output)
    }
}

pub fn create_embedding(
    vocab_size: usize,
    embedding_dim: usize,
    device: &Device,
) -> Result<Embedding> {
    let var_builder: VarBuilder = VarBuilder::zeros(DType::F32, device);

    let embedding: Embedding = embedding(vocab_size, embedding_dim, var_builder)?;

    Ok(embedding)
}

pub fn apply_embedding(embedding: &Embedding, token_ids: &Tensor) -> Result<Tensor> {
    let output: Tensor = embedding.forward(token_ids)?;

    Ok(output)
}

pub fn masked_softmax(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
    if scores.dims().len() != 3 {
        anyhow::bail!(
            "Expected scores with shape [batch, sequence, sequence], got {:?}",
            scores.dims()
        );
    }

    if mask.dims().len() != 2 {
        anyhow::bail!(
            "Expected mask with shape [sequence, sequence], got {:?}",
            mask.dims()
        );
    }

    let sequence_length: usize = scores.dims()[1];

    if scores.dims()[2] != sequence_length {
        anyhow::bail!("Attention scores must be square");
    }

    if mask.dims() != [sequence_length, sequence_length] {
        anyhow::bail!(
            "Mask shape {:?} does not match attention scores",
            mask.dims()
        );
    }

    let masked_scores: Tensor = scores.broadcast_add(mask)?;

    let probabilities: Tensor = candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;

    Ok(probabilities)
}

pub fn apply_attention_weights(attention_weights: &Tensor, value: &Tensor) -> Result<Tensor> {
    if attention_weights.dims().len() != 3 {
        anyhow::bail!(
            "Attention weights must have shape [batch, sequence, sequence], got {:?}",
            attention_weights.dims()
        );
    }

    if value.dims().len() != 3 {
        anyhow::bail!(
            "Value must have shape [batch, sequence, embedding], got {:?}",
            value.dims()
        );
    }

    if attention_weights.dims()[0] != value.dims()[0] {
        anyhow::bail!("Batch sizes do not match");
    }

    if attention_weights.dims()[2] != value.dims()[1] {
        anyhow::bail!("Attention sequence dimension does not match value sequence dimension");
    }

    let output: Tensor = attention_weights.matmul(value)?;

    Ok(output)
}

pub struct CausalSelfAttention {
    qkv: QkvProjection,
    embedding_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(embedding_dim: usize, var_builder: VarBuilder) -> Result<Self> {
        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        let qkv: QkvProjection = QkvProjection::new(embedding_dim, var_builder.pp("qkv"))?;

        Ok(Self { qkv, embedding_dim })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        if input.dims()[2] != self.embedding_dim {
            anyhow::bail!(
                "Input embedding dimension {} does not match expected embedding dimension {}",
                input.dims()[2],
                self.embedding_dim
            );
        }

        let (query, key, value): (Tensor, Tensor, Tensor) = self.qkv.forward(input)?;

        let scores: Tensor = scaled_attention_scores(&query, &key)?;

        let sequence_length: usize = input.dims()[1];

        let mask: Tensor = create_causal_mask(sequence_length, input.device())?;

        let attention_weights: Tensor = masked_softmax(&scores, &mask)?;

        let output: Tensor = apply_attention_weights(&attention_weights, &value)?;

        Ok(output)
    }
}

pub fn create_causal_mask(sequence_length: usize, device: &Device) -> Result<Tensor> {
    if sequence_length == 0 {
        anyhow::bail!("Sequence length cannot be zero");
    }

    let mut mask_values: Vec<f32> = Vec::with_capacity(sequence_length * sequence_length);

    for row in 0..sequence_length {
        for column in 0..sequence_length {
            if column <= row {
                mask_values.push(0.0);
            } else {
                mask_values.push(f32::NEG_INFINITY);
            }
        }
    }

    let mask: Tensor =
        Tensor::new(mask_values.as_slice(), device)?.reshape((sequence_length, sequence_length))?;

    Ok(mask)
}

pub struct MultiHeadAttention {
    qkv: QkvProjection,
    output_projection: Linear,
    embedding_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(embedding_dim: usize, num_heads: usize, var_builder: VarBuilder) -> Result<Self> {
        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        if num_heads == 0 {
            anyhow::bail!("Number of heads cannot be zero");
        }

        if embedding_dim % num_heads != 0 {
            anyhow::bail!(
                "Embedding dimension {} must be divisible by number of heads {}",
                embedding_dim,
                num_heads
            );
        }

        let head_dim: usize = embedding_dim / num_heads;

        let qkv: QkvProjection = QkvProjection::new(embedding_dim, var_builder.pp("qkv"))?;

        let output_projection: Linear =
            linear(embedding_dim, embedding_dim, var_builder.pp("output"))?;

        Ok(Self {
            qkv,
            output_projection,
            embedding_dim,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        if input.dims()[2] != self.embedding_dim {
            anyhow::bail!(
                "Input embedding dimension {} does not match expected embedding dimension {}",
                input.dims()[2],
                self.embedding_dim
            );
        }

        let batch_size: usize = input.dims()[0];

        let sequence_length: usize = input.dims()[1];

        let (query, key, value): (Tensor, Tensor, Tensor) = self.qkv.forward(input)?;

        let query: Tensor = query
            .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let key: Tensor = key
            .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let value: Tensor = value
            .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let key_transposed: Tensor = key.transpose(2, 3)?.contiguous()?;

        let scores: Tensor = query.matmul(&key_transposed)?;

        let scale: f64 = (self.head_dim as f64).sqrt();

        let scores: Tensor = scores.affine(1.0 / scale, 0.0)?;

        let mask: Tensor = create_causal_mask(sequence_length, input.device())?;

        let mask: Tensor = mask.unsqueeze(0)?.unsqueeze(0)?;

        let masked_scores: Tensor = scores.broadcast_add(&mask)?;

        let attention_weights: Tensor =
            candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;

        let attended: Tensor = attention_weights.matmul(&value)?;

        let attended: Tensor = attended.transpose(1, 2)?.contiguous()?;

        let attended: Tensor =
            attended.reshape((batch_size, sequence_length, self.embedding_dim))?;

        let output: Tensor = self.output_projection.forward(&attended)?;

        Ok(output)
    }
}

pub struct FeedForward {
    first: Linear,
    second: Linear,
    embedding_dim: usize,
    hidden_dim: usize,
}

impl FeedForward {
    pub fn new(embedding_dim: usize, hidden_dim: usize, var_builder: VarBuilder) -> Result<Self> {
        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        if hidden_dim == 0 {
            anyhow::bail!("Hidden dimension cannot be zero");
        }

        let first: Linear = linear(embedding_dim, hidden_dim, var_builder.pp("first"))?;

        let second: Linear = linear(hidden_dim, embedding_dim, var_builder.pp("second"))?;

        Ok(Self {
            first,
            second,
            embedding_dim,
            hidden_dim,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        if input.dims()[2] != self.embedding_dim {
            anyhow::bail!(
                "Input embedding dimension {} does not match expected embedding dimension {}",
                input.dims()[2],
                self.embedding_dim
            );
        }

        let hidden: Tensor = self.first.forward(input)?;

        let hidden: Tensor = candle_nn::ops::silu(&hidden)?;

        let output: Tensor = self.second.forward(&hidden)?;

        Ok(output)
    }
}

pub struct LayerNorm {
    layer_norm: candle_nn::LayerNorm,
    embedding_dim: usize,
}

impl LayerNorm {
    pub fn new(embedding_dim: usize, eps: f64, var_builder: VarBuilder) -> Result<Self> {
        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        if eps <= 0.0 {
            anyhow::bail!("Epsilon must be greater than zero");
        }

        let layer_norm: candle_nn::LayerNorm =
            candle_nn::layer_norm(embedding_dim, eps, var_builder)?;

        Ok(Self {
            layer_norm,
            embedding_dim,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        if input.dims()[2] != self.embedding_dim {
            anyhow::bail!(
                "Input embedding dimension {} does not match expected embedding dimension {}",
                input.dims()[2],
                self.embedding_dim
            );
        }

        let output: Tensor = self.layer_norm.forward(input)?;

        Ok(output)
    }
}

pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: MultiHeadAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
    embedding_dim: usize,
}

impl TransformerBlock {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        eps: f64,
        var_builder: VarBuilder,
    ) -> Result<Self> {
        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        if num_heads == 0 {
            anyhow::bail!("Number of heads cannot be zero");
        }

        if hidden_dim == 0 {
            anyhow::bail!("Hidden dimension cannot be zero");
        }

        let attention_norm: LayerNorm =
            LayerNorm::new(embedding_dim, eps, var_builder.pp("attention_norm"))?;

        let attention: MultiHeadAttention =
            MultiHeadAttention::new(embedding_dim, num_heads, var_builder.pp("attention"))?;

        let feed_forward_norm: LayerNorm =
            LayerNorm::new(embedding_dim, eps, var_builder.pp("feed_forward_norm"))?;

        let feed_forward: FeedForward =
            FeedForward::new(embedding_dim, hidden_dim, var_builder.pp("feed_forward"))?;

        Ok(Self {
            attention_norm,
            attention,
            feed_forward_norm,
            feed_forward,
            embedding_dim,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.dims().len() != 3 {
            anyhow::bail!(
                "Expected input with shape [batch, sequence, embedding], got {:?}",
                input.dims()
            );
        }

        if input.dims()[2] != self.embedding_dim {
            anyhow::bail!(
                "Input embedding dimension {} does not match expected embedding dimension {}",
                input.dims()[2],
                self.embedding_dim
            );
        }

        let normalized_input: Tensor = self.attention_norm.forward(input)?;

        let attention_output: Tensor = self.attention.forward(&normalized_input)?;

        let attention_residual: Tensor = input.add(&attention_output)?;

        let normalized_attention: Tensor = self.feed_forward_norm.forward(&attention_residual)?;

        let feed_forward_output: Tensor = self.feed_forward.forward(&normalized_attention)?;

        let output: Tensor = attention_residual.add(&feed_forward_output)?;

        Ok(output)
    }
}

pub struct OrangeAideModel {
    token_embedding: Embedding,
    positional_embedding: PositionalEmbedding,
    transformer_blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    language_model_head: Linear,
    vocab_size: usize,
    embedding_dim: usize,
    sequence_length: usize,
}

impl OrangeAideModel {
    pub fn new(
        vocab_size: usize,
        sequence_length: usize,
        embedding_dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        num_layers: usize,
        eps: f64,
        var_builder: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        if vocab_size == 0 {
            anyhow::bail!("Vocabulary size cannot be zero");
        }

        if sequence_length == 0 {
            anyhow::bail!("Sequence length cannot be zero");
        }

        if embedding_dim == 0 {
            anyhow::bail!("Embedding dimension cannot be zero");
        }

        if num_heads == 0 {
            anyhow::bail!("Number of heads cannot be zero");
        }

        if hidden_dim == 0 {
            anyhow::bail!("Hidden dimension cannot be zero");
        }

        if num_layers == 0 {
            anyhow::bail!("Number of layers cannot be zero");
        }

        let token_embedding: Embedding =
            embedding(vocab_size, embedding_dim, var_builder.pp("token_embedding"))?;

        let positional_embedding: PositionalEmbedding = PositionalEmbedding::new_with_var_builder(
            sequence_length,
            embedding_dim,
            var_builder.pp("positional_embedding"),
        )?;

        let mut transformer_blocks: Vec<TransformerBlock> = Vec::with_capacity(num_layers);

        for layer_index in 0..num_layers {
            let block: TransformerBlock = TransformerBlock::new(
                embedding_dim,
                num_heads,
                hidden_dim,
                eps,
                var_builder.pp(format!("block_{}", layer_index)),
            )?;

            transformer_blocks.push(block);
        }

        let final_norm: LayerNorm =
            LayerNorm::new(embedding_dim, eps, var_builder.pp("final_norm"))?;

        let language_model_head: Linear = linear(
            embedding_dim,
            vocab_size,
            var_builder.pp("language_model_head"),
        )?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            transformer_blocks,
            final_norm,
            language_model_head,
            vocab_size,
            embedding_dim,
            sequence_length,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        if token_ids.dims().len() != 2 {
            anyhow::bail!(
                "Expected token IDs with shape [batch, sequence], got {:?}",
                token_ids.dims()
            );
        }

        let sequence_length: usize = token_ids.dims()[1];

        if sequence_length == 0 {
            anyhow::bail!("Input sequence cannot be zero");
        }

        if sequence_length > self.sequence_length {
            anyhow::bail!(
                "Input sequence length {} exceeds maximum sequence length {}",
                sequence_length,
                self.sequence_length
            );
        }

        let token_embeddings: Tensor = apply_embedding(&self.token_embedding, token_ids)?;

        let mut hidden_states: Tensor = self.positional_embedding.forward(&token_embeddings)?;

        for block in &self.transformer_blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        hidden_states = self.final_norm.forward(&hidden_states)?;

        let logits: Tensor = self.language_model_head.forward(&hidden_states)?;

        Ok(logits)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}
