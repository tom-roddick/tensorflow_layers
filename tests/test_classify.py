import tensorflow as tf
import unittest
from .. import classify

class TestSpatialCrossEntropy(unittest.TestCase):

    def test_basic(self):

        logits = tf.constant(
            [[[[1., 5., 1.],
               [2., 4., 2.],
               [3., 3., 3.],
               [4., 2., 4.],
               [5., 1., 5.]]]])

        labels = tf.constant([[[1, 1, 2, 0, 2]]])

        xent = classify.spatial_cross_entropy(logits, labels)

        with tf.Session() as sess:

            print "Input"
            print(logits.eval())
            print "Labels"
            print(labels.eval())

            loss = sess.run(xent)
            print("\nLoss = {}".format(loss))

            self.assertTrue(loss >= 1.2308 and loss <= 1.2317)
