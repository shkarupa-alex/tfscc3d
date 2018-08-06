from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.training import monitored_session
from .. import sequence_categorical_column_with_hash_bucket, sequence_categorical_column_with_vocabulary_list


def _assert_sparse_tensor_value(test_case, expected, actual):
    _assert_sparse_tensor_indices_shape(test_case, expected, actual)

    test_case.assertEqual(np.array(expected.values).dtype, np.array(actual.values).dtype)
    test_case.assertAllEqual(expected.values, actual.values)


def _assert_sparse_tensor_indices_shape(test_case, expected, actual):
    test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
    test_case.assertAllEqual(expected.indices, actual.indices)

    test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
    test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class SequenceCategoricalColumnWithHashBucketTest(tf.test.TestCase):
    def test_get_sparse_tensors2d(self):
        column = sequence_categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
        inputs = tf.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=('omar', 'stringer', 'marlo'),
            dense_shape=(2, 2))
        expected_sparse_ids = tf.SparseTensorValue(
            indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            # Ignored to avoid hash dependence in test.
            values=np.array((0, 0, 0), dtype=np.int64),
            dense_shape=(2, 2, 1))

        id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

        self.assertIsNone(id_weight_pair.weight_tensor)
        with monitored_session.MonitoredSession() as sess:
            _assert_sparse_tensor_indices_shape(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))

    def test_get_sparse_tensors3d(self):
        column = sequence_categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
        inputs = tf.SparseTensorValue(
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
            values=('marlo', 'omar', 'skywalker', 'omar'),
            dense_shape=(2, 2, 2))
        expected_sparse_ids = tf.SparseTensorValue(
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
            # Ignored to avoid hash dependence in test.
            values=np.array((0, 0, 0, 0), dtype=np.int64),
            dense_shape=(2, 2, 2))

        id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

        self.assertIsNone(id_weight_pair.weight_tensor)
        with monitored_session.MonitoredSession() as sess:
            _assert_sparse_tensor_indices_shape(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))


class SequenceCategoricalColumnWithVocabularyListTest(tf.test.TestCase):
    def test_get_sparse_tensors2d(self):
        column = sequence_categorical_column_with_vocabulary_list(
            key='aaa',
            vocabulary_list=('omar', 'stringer', 'marlo'))
        inputs = tf.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=('marlo', 'skywalker', 'omar'),
            dense_shape=(2, 2))
        expected_sparse_ids = tf.SparseTensorValue(
            indices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=(2, 2, 1))

        id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

        self.assertIsNone(id_weight_pair.weight_tensor)
        with monitored_session.MonitoredSession() as sess:
            _assert_sparse_tensor_indices_shape(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))
            _assert_sparse_tensor_value(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))

    def test_get_sparse_tensors3d(self):
        column = sequence_categorical_column_with_vocabulary_list(
            key='aaa',
            vocabulary_list=('omar', 'stringer', 'marlo'))
        inputs = tf.SparseTensorValue(
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
            values=('marlo', 'omar', 'skywalker', 'omar'),
            dense_shape=(2, 2, 2))
        expected_sparse_ids = tf.SparseTensorValue(
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
            values=np.array((2, 0, -1, 0), dtype=np.int64),
            dense_shape=(2, 2, 2))

        id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

        self.assertIsNone(id_weight_pair.weight_tensor)
        with monitored_session.MonitoredSession() as sess:
            _assert_sparse_tensor_indices_shape(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))
            _assert_sparse_tensor_value(self, expected_sparse_ids, id_weight_pair.id_tensor.eval(session=sess))


class SequenceEmbeddingColumnTest(tf.test.TestCase):
    def test_get_sequence_dense_tensor(self):
        vocabulary_size = 3
        sparse_input = tf.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (3, 0, 0)),
            values=(2, 2, 0, 1, 1),
            dense_shape=(4, 2, 2))

        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info):
            self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            return embedding_values

        expected_lookups = [
            # example 0, ids [2]
            [[6., 10.], [0., 0.]],
            # example 1, ids [0, 1]
            [[3., 5.], [1., 2.]],
            # example 2, ids []
            [[0., 0.], [0., 0.]],
            # example 3, ids [1]
            [[1., 2.], [0., 0.]],
        ]

        categorical_column = sequence_categorical_column_with_hash_bucket(
            key='aaa', hash_bucket_size=vocabulary_size, dtype=tf.int32)
        embedding_column = fc.embedding_column(
            categorical_column, dimension=embedding_dimension, combiner='sum', initializer=_initializer)

        embedding_lookup, _ = embedding_column._get_sequence_dense_tensor(
            _LazyBuilder({'aaa': sparse_input}))

        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.assertItemsEqual(
            ('embedding_weights:0',), tuple([v.name for v in global_vars]))
        with monitored_session.MonitoredSession() as sess:
            self.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
            self.assertAllEqual(expected_lookups, embedding_lookup.eval(session=sess))

    def test_sequence_length(self):
        vocabulary_size = 3
        sparse_input = tf.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)),
            values=(2, 2, 0, 1),
            dense_shape=(2, 2, 2))
        expected_sequence_length = [1, 2]

        categorical_column = sequence_categorical_column_with_hash_bucket(
            key='aaa', hash_bucket_size=vocabulary_size, dtype=tf.int32)
        embedding_column = fc.embedding_column(categorical_column, dimension=2)

        _, sequence_length = embedding_column._get_sequence_dense_tensor(
            _LazyBuilder({'aaa': sparse_input}))

        with monitored_session.MonitoredSession() as sess:
            sequence_length = sess.run(sequence_length)
            self.assertAllEqual(expected_sequence_length, sequence_length)
            self.assertEqual(np.int64, sequence_length.dtype)

    def test_sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        vocabulary_size = 3
        sparse_input = tf.SparseTensorValue(
            # example 0, ids []
            # example 1, ids [2]
            # example 2, ids [0, 1]
            # example 3, ids []
            # example 4, ids [1]
            # example 5, ids []
            indices=((1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 1, 0), (4, 0, 0)),
            values=(2, 2, 0, 1, 1),
            dense_shape=(6, 2, 2))
        expected_sequence_length = [0, 1, 2, 0, 1, 0]

        categorical_column = sequence_categorical_column_with_hash_bucket(
            key='aaa', hash_bucket_size=vocabulary_size, dtype=tf.int32)
        embedding_column = fc.embedding_column(categorical_column, dimension=2)

        _, sequence_length = embedding_column._get_sequence_dense_tensor(
            _LazyBuilder({'aaa': sparse_input}))

        with monitored_session.MonitoredSession() as sess:
            self.assertAllEqual(
                expected_sequence_length, sequence_length.eval(session=sess))


if __name__ == '__main__':
    tf.test.main()
