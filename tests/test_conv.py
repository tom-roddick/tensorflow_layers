import tensorflow as tf
import unittest
from .. import conv

class TestSpatialMaxPooling(unittest.TestCase):

    def test_range3d(self):
        rng = tf.to_float(tf.range(1, 33))
        inp = tf.reshape(rng,[1, 4, 4, 2])

        target = tf.constant([[[11., 12.], [15., 16.]], [[27., 28.], [31., 32.]]])
        output, indices = conv.spatial_max_pooling(inp, 2, 2)

        assert_op = tf.equal(output, target)
        print_inp = tf.Print(assert_op, [inp], message="Input: ", summarize=16)
        print_out = tf.Print(print_inp, [output], message="Output: ", summarize=8)
        print_tar = tf.Print(print_out, [target], message="Target: ", summarize=8)
        print_inds = tf.Print(print_tar, [indices], message="Indices: ", summarize=8)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            result = sess.run(print_inds)

            self.assertTrue(result.all())

class TestSpatialUnpooling(unittest.TestCase):

    def test_unpooling(self):

        rng = tf.to_float(tf.range(1, 33))
        inp = tf.reshape(rng,[1, 4, 4, 2])
        target = tf.constant([[[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0],[11., 12.],[0, 0],[15.,16.]],
            [[0, 0],[0, 0],[0, 0], [0, 0]],
            [[0, 0], [27., 28.], [0, 0], [31., 32.]]])

        pooled, indices = conv.spatial_max_pooling(inp, 2, 2)

        output = conv.spatial_unpooling(pooled, indices, 2, 2)

        assert_op = tf.equal(output, target)
        print_inp = tf.Print(assert_op, [inp], message="Input: ", summarize=32)
        print_out = tf.Print(print_inp, [output], message="Output: ", summarize=32)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            result = sess.run(print_out)

            self.assertTrue(result.all())

if __name__ == '__main__':
    unittest.main()
