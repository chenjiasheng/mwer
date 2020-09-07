import tensorflow as tf


def mwer_loss(
        candidate_seqs,
        candidate_seq_lens,
        candidate_seq_cnts,
        candidate_seq_edit_errors,
        candidate_seq_logprobs
):
    """Computes the mWER (minimum WER) Loss.

    This op implements the mWER loss as presented in the article:

    [Rohit Prabhavalkar etc.
    MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED
    SEQUENCE-TO-SEQUENCE MODELS](https://arxiv.org/pdf/1712.01818.pdf)

    Input requirements:

    Notations:
      B: batch size
      N: the number of candidate sequences (i.e. hypothesis sequences) plus 1.
         the last sequence is treated as the ground truth and used to compute ce loss.
      U: max length of candidate sequences including SOS (but not EOS).
      V: vocabulary size; number of tokens.

    Args:
      candidate_seqs: An `int32` `Tensor` with shape (B, N, U + 1).
        `candidate_seqs[b, n, u]` means u-th token-id of n-th candidate sequence (including SOS and EOS) of b-th sample.
      candidate_seq_lens: An `int32` `Tensor` with shape (B, N,).
        actual lengths of each candidate sequence, including SOS and EOS.
        candidate_seq_lens[n] <= U + 1 for n in range(N)
      candidate_seq_edit_errors: An `float32` `Tensor` with shape (B, N,).
        the edit distance error for each candidate sequence.
      candidate_seq_logprobs: An `float32` `Tensor` with shape (B, N, U, V).
        `candidate_seq_logprobs[b, n, u, v]` the log prob of being token v for the u-th token
        of n-th candidate sequence (not including SOS) of b-th sample.
      candidate_seq_cnts: An `int32` `Tensor` with shape (B, 1).
        candidate_seq_cnts[b] <= N for all b in range(B). denotes the real number effective candidate sequences.
        because sometimes there's not enough N candidates since the acoustic encoder is very sure
        about its few hypothesis.

    Returns:
      weighted_relative_edit_errors:
        A 1-D `float` `Tensor`, size `[B]`, a batch of mWER loss.
      ce_loss:
        A 1-D `float` `Tensor`, size `[B]`, a batch of CE loss.
      rescore_wer:
        A 1-D `float` `Tensor`, size `[B]`, WER of the top-1 rescored sequences.
        This output is only for metric/evaluate and won't be back propagated.
    """
    int_shape = candidate_seq_logprobs.get_shape().as_list()
    shape = tf.shape(candidate_seq_logprobs)

    B = int_shape[0] or shape[0]
    N = int_shape[1]
    U = int_shape[2] or shape[2]
    V = int_shape[3]

    flatten_logprobs = tf.reshape(candidate_seq_logprobs, (-1, V))  # (B * N * U, V)
    flatten_tokens = tf.reshape(candidate_seqs[:, :, 1:], shape=(-1,))  # (B * N * U,)
    indices = tf.transpose(tf.stack([tf.range(B * N * U), flatten_tokens]))  # (B * N * U, 2)
    flatten_logprobs = tf.gather_nd(flatten_logprobs, indices)  # (B * N * U,)
    token_logprobs = tf.reshape(flatten_logprobs, (B * N, U))  # (B * N, U)

    token_mask = tf.sequence_mask(
        tf.reshape(candidate_seq_lens - 1, shape=(-1,)),
        maxlen=U,
        dtype=tf.dtypes.float32
    )  # (B * N, U)
    masked_token_logprobs = token_logprobs * token_mask  # (B * N, U)
    masked_token_logprobs = tf.reshape(masked_token_logprobs, (B, N, U))  # (B, N, U)
    seq_logprobs = tf.reduce_sum(masked_token_logprobs, axis=-1)  # (B, N)

    def softmax_with_mask(logits, mask):
        mask = tf.cast(mask, tf.dtypes.float32)
        logits -= 10000.0 * (1.0 - mask)
        ai = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
        softmax_result = ai / (tf.reduce_sum(ai, axis=1, keepdims=True) + 1e-10)
        return softmax_result

    # mask out the padding seqs and the final ground truth seq.
    seq_mask = tf.sequence_mask(
        tf.reshape(candidate_seq_cnts, shape=(-1,)),
        maxlen=N,
        dtype=tf.dtypes.float32
    )  # (B, N)
    renormalized_seq_probs = softmax_with_mask(seq_logprobs, seq_mask)  # (B, N)

    masked_edit_errors = seq_mask * candidate_seq_edit_errors  # (B, N)
    avg_edit_errors = tf.reduce_sum(masked_edit_errors, axis=-1, keepdims=True) / tf.cast(candidate_seq_cnts, 'float32')  # (B, 1)
    relative_edit_errors = seq_mask * (masked_edit_errors - tf.tile(avg_edit_errors, (1, N)))

    weighted_relative_edit_errors = tf.reduce_sum(renormalized_seq_probs * relative_edit_errors, axis=-1)  # (B,)
    # the last seq of each sample is used to calculate CE loss
    ce_loss = -seq_logprobs[:, -1]

    top1_seq_indices = tf.argmax(renormalized_seq_probs, axis=-1, output_type=tf.dtypes.int32)  # (B,)
    indices = tf.transpose(tf.stack([tf.range(B), top1_seq_indices]))  # (B, 2)
    chosen_seq_edit_errors = tf.gather_nd(masked_edit_errors, indices)  # (B,)
    ground_seq_len = candidate_seq_lens[:, -1] - 2  # (B,)
    rescore_wer = chosen_seq_edit_errors / tf.cast(ground_seq_len, tf.dtypes.float32)
    rescore_wer = tf.stop_gradient(rescore_wer)

    return [weighted_relative_edit_errors, ce_loss, rescore_wer]
