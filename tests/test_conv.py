import tensorflow as tf
import unittest
import conv

class TestSpatialMaxPooling(unittest.TestCase):

    def test_range3d(self):
        rng = tf.to_float(tf.range(1, 32))
        inp = tf.reshape(rng,[1, 4, 4, 2])

        target = tf.constant([[[6, 22], [8, 24]], [[14, 30], [16, 32]]])
        output = conv.spatial_max_pooling(inp, 2, 2)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(output)

            self.assertTrue(output==target)

            print("Output")
            print(output)
            print("Target")
            print(target)

if __name__ == '__main__':
    unittest.main()
