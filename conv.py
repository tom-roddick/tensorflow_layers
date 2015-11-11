import tensorflow as tf

def spatial_conv(inputs, noutput_chls, kW, kH, dW=1, dH=1,
        padding = "SAME", init = 1.0, name=None):
        """Convolutional layer which applies a 2D filter over an image.

        Args:
            inputs: 4-D tensor with dimensions [batch_size, height,
                width, ninput_chls].
            noutput_chls: Number of output channels
            kW: Kernel width
            kH: Kernel height
            dW: Horizontal stride (default 1)
            dH: Vertical stride (default 1)
            padding: A string value of either 'SAME' or 'VALID'
                    'SAME': 0 padding
                    'VALID': (kW-1)/2 padding
            init: Initialise weights with standard deviation of init
            name: Name scope for the module

        Returns:
            output: 4-D feature map with dimensions [batch_size, height,
                width, noutput_chls]"""

    ninput_chls = inputs.get_shape().as_list()[3]

    weights = tf.Variable(
        tf.truncated_normal([kH, kW, ninput_chls, noutput_chls], stddev = init),
        name = 'weights')
    biases = tf.Variable(tf.zeros(noutput_chls), name='biases')

    conv = tf.nn.conv2d(inputs, weights, [1, dH, dW, 1], padding = padding)
