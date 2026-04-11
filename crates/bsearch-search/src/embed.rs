use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;

/// Loads an ONNX model and tokenizer, then uses them to encode text into
/// 384-dimensional embeddings.
pub struct Embedder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl Embedder {
    /// Load the ONNX model and tokenizer from a directory.
    /// Expects `model.onnx` and `tokenizer.json` in `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        anyhow::ensure!(
            model_path.exists(),
            "ONNX model not found at {}. Run `bsearch export-model` first.",
            model_path.display()
        );
        anyhow::ensure!(
            tokenizer_path.exists(),
            "Tokenizer not found at {}. Run `bsearch export-model` first.",
            tokenizer_path.display()
        );

        let session = Session::builder()
            .context("Failed to create ONNX Runtime session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Embedder { session, tokenizer })
    }

    /// Encode a single text into a 384-dimensional embedding vector.
    pub fn encode(&mut self, text: &str) -> Result<[f32; 384]> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenisation failed: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = input_ids.len();

        let ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids.into_boxed_slice()))
            .context("input_ids tensor")?;
        let mask_tensor =
            Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))
                .context("attention_mask tensor")?;
        let type_tensor =
            Tensor::from_array(([1usize, seq_len], token_type_ids.into_boxed_slice()))
                .context("token_type_ids tensor")?;

        let outputs = self.session.run(ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_tensor,
        })?;

        // Output: last_hidden_state with shape (1, seq_len, 384)
        let output_view = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract output tensor")?;

        // Reshape from (1, seq_len, 384) to (seq_len, 384)
        let hidden_state: Array2<f32> = output_view
            .into_shape_with_order((seq_len, 384))
            .context("Failed to reshape hidden state")?
            .to_owned();

        // Build attention mask as f32 for mean pooling
        let mask_f32: Array1<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as f32)
            .collect();

        let pooled = mean_pool(&hidden_state, &mask_f32);
        let normalised = l2_normalise(&pooled);

        let mut result = [0.0f32; 384];
        result.copy_from_slice(
            normalised
                .as_slice()
                .context("Embedding is not contiguous")?,
        );
        Ok(result)
    }
}

/// Mean-pool token embeddings, masking padding tokens.
///
/// `hidden_state` has shape (seq_len, hidden_dim).
/// `attention_mask` has shape (seq_len,) with 1.0 for real tokens and 0.0 for padding.
pub fn mean_pool(hidden_state: &Array2<f32>, attention_mask: &Array1<f32>) -> Array1<f32> {
    let hidden_dim = hidden_state.ncols();
    let mut sum = Array1::<f32>::zeros(hidden_dim);
    let mut mask_sum: f32 = 0.0;

    for (i, mask_val) in attention_mask.iter().enumerate() {
        if *mask_val > 0.0 {
            sum += &(hidden_state.row(i).to_owned() * *mask_val);
            mask_sum += mask_val;
        }
    }

    if mask_sum > 0.0 {
        sum /= mask_sum;
    }
    sum
}

/// L2-normalise a vector, returning a unit vector.
pub fn l2_normalise(v: &Array1<f32>) -> Array1<f32> {
    let norm = v.dot(v).sqrt();
    if norm > 0.0 { v / norm } else { v.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_pool_simple() {
        let hidden = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let mask = array![1.0, 1.0, 1.0];
        let result = mean_pool(&hidden, &mask);
        assert_eq!(result, array![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mean_pool_with_padding() {
        let hidden = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99.0, 99.0, 99.0, 99.0,
            ],
        )
        .unwrap();
        let mask = array![1.0, 1.0, 0.0];
        let result = mean_pool(&hidden, &mask);
        assert_eq!(result, array![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_l2_normalise() {
        let v = array![3.0, 4.0];
        let normed = l2_normalise(&v);
        let expected = array![0.6, 0.8];
        for (a, b) in normed.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_l2_normalise_unit_length() {
        let v = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normed = l2_normalise(&v);
        let length: f32 = normed.dot(&normed).sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalise_zero_vector() {
        let v = array![0.0, 0.0, 0.0];
        let normed = l2_normalise(&v);
        assert_eq!(normed, array![0.0, 0.0, 0.0]);
    }
}
