from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.feature_column import feature_column as fc
import tensorflow as tf


def sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string):
    """A sequence of categorical terms where ids are set by hashing.

    Pass this to `embedding_column` or `indicator_column` to convert sequence
    categorical data into dense representation for input to sequence NN, such as
    RNN.

    Example:

    ```python
    tokens = sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=1000)
    tokens_embedding = embedding_column(tokens, dimension=10)
    columns = [tokens_embedding]

    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    input_layer, sequence_length = sequence_input_layer(features, columns)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    outputs, state = tf.nn.dynamic_rnn(
        rnn_cell, inputs=input_layer, sequence_length=sequence_length)
    ```

    Args:
      key: A unique string identifying the input feature.
      hash_bucket_size: An int > 1. The number of buckets.
      dtype: The type of features. Only string and integer types are supported.

    Returns:
      A `_SequenceCategoricalColumn`.

    Raises:
      ValueError: `hash_bucket_size` is not greater than 1.
      ValueError: `dtype` is neither string nor integer.
    """
    return _SequenceCategoricalColumn(fc.categorical_column_with_hash_bucket(
        key=key,
        hash_bucket_size=hash_bucket_size,
        dtype=dtype))


def sequence_categorical_column_with_vocabulary_list(
        key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0):
    """A sequence of categorical terms where ids use an in-memory list.

    Pass this to `embedding_column` or `indicator_column` to convert sequence
    categorical data into dense representation for input to sequence NN, such as
    RNN.

    Example:

    ```python
    colors = sequence_categorical_column_with_vocabulary_list(
        key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
        num_oov_buckets=2)
    colors_embedding = embedding_column(colors, dimension=3)
    columns = [colors_embedding]

    features = tf.parse_example(..., features=make_parse_example_spec(columns))
    input_layer, sequence_length = sequence_input_layer(features, columns)

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    outputs, state = tf.nn.dynamic_rnn(
        rnn_cell, inputs=input_layer, sequence_length=sequence_length)
    ```

    Args:
      key: A unique string identifying the input feature.
      vocabulary_list: An ordered iterable defining the vocabulary. Each feature
        is mapped to the index of its value (if present) in `vocabulary_list`.
        Must be castable to `dtype`.
      dtype: The type of features. Only string and integer types are supported.
        If `None`, it will be inferred from `vocabulary_list`.
      default_value: The integer ID value to return for out-of-vocabulary feature
        values, defaults to `-1`. This can not be specified with a positive
        `num_oov_buckets`.
      num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
        buckets. All out-of-vocabulary inputs will be assigned IDs in the range
        `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
        hash of the input value. A positive `num_oov_buckets` can not be specified
        with `default_value`.

    Returns:
      A `_SequenceCategoricalColumn`.

    Raises:
      ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
      ValueError: `num_oov_buckets` is a negative integer.
      ValueError: `num_oov_buckets` and `default_value` are both specified.
      ValueError: if `dtype` is not integer or string.
    """
    return _SequenceCategoricalColumn(fc.categorical_column_with_vocabulary_list(
        key=key,
        vocabulary_list=vocabulary_list,
        dtype=dtype,
        default_value=default_value,
        num_oov_buckets=num_oov_buckets))


class _SequenceCategoricalColumn(fc._SequenceCategoricalColumn):
    """Represents sequences of categorical data."""

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        sparse_tensors = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
        id_tensor = sparse_tensors.id_tensor
        weight_tensor = sparse_tensors.weight_tensor
        # Expands final dimension, so that embeddings are not combined during
        # embedding lookup.
        check_id_rank_low = tf.assert_greater_equal(
            tf.rank(id_tensor), 2,
            data=['Column {} expected ID tensor of rank 2 or 3. '.format(self.name),
                  'id_tensor shape: ', tf.shape(id_tensor)])
        check_id_rank_high = tf.assert_less_equal(
            tf.rank(id_tensor), 3,
            data=['Column {} expected ID tensor of rank 2 or 3. '.format(self.name),
                  'id_tensor shape: ', tf.shape(id_tensor)])
        with tf.control_dependencies([check_id_rank_low, check_id_rank_high]):
            id_tensor_shape = tf.concat([id_tensor.dense_shape, [1]], axis=0)
            id_tensor_shape = tf.slice(id_tensor_shape, [0], [3])
            id_tensor = tf.sparse_reshape(id_tensor, shape=id_tensor_shape)
        if weight_tensor is not None:
            check_weight_rank_low = tf.assert_greater_equal(
                tf.rank(weight_tensor), 2,
                data=['Column {} expected weight tensor of rank 2 or 3.'.format(self.name),
                      'weight_tensor shape:', tf.shape(weight_tensor)])
            check_weight_rank_high = tf.assert_less_equal(
                tf.rank(weight_tensor), 3,
                data=['Column {} expected weight tensor of rank 2 or 3.'.format(self.name),
                      'weight_tensor shape:', tf.shape(weight_tensor)])
            with tf.control_dependencies([check_weight_rank_low, check_weight_rank_high]):
                weight_tensor_shape = tf.concat([weight_tensor.dense_shape, [1]], axis=0)
                weight_tensor_shape = tf.slice(weight_tensor_shape, [0], [3])
                weight_tensor = tf.sparse_reshape(weight_tensor, shape=weight_tensor_shape)
        return fc._CategoricalColumn.IdWeightPair(id_tensor, weight_tensor)
