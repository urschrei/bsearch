use ndarray::{Array1, Array2};

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
