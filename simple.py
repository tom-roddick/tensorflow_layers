import tensorflow as tf

def linear(inputs, noutput_chls, init = 1.0, name = None):
    """Fully connected layer which applies a simple linear transformation
    to the inputs.

    Args:
        inputs: 3-D tensor with dimensions [batch_size, N, ninput_chls].
        noutput_chls: Number of output channels
        init: Initialise weights with standard deviation of init
        name: Name scope for the module

    Returns:
        output: 3-D tensor with dimensions [batch_size, N, noutput_chls]"""

    ninput_chls = inputs.get_shape().as_list()[2]

    weights = tf.Variable(
        tf.truncated_normal([ninput_chls, noutput_chls],
            stddev = init),
        name = 'weights')

    biases = tf.Variable(tf.zeros([noutput_chls]),
        name = 'biases')

    return tf.matmul(inputs, weights) + biases
