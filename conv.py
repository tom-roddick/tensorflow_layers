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
        padding: A string value of either 'SAME' or 'VALID' (default 'SAME')
                'SAME': 0 padding
                'VALID': (kW-1)/2 padding
        init: Initialise weights with standard deviation of init
        name: Name scope for the module

    Returns:
        output: 4-D feature map with dimensions [batch_size, h_output,
            w_output, noutput_chls]"""

    ninput_chls = inputs.get_shape().as_list()[3]

    weights = tf.Variable(
        tf.truncated_normal([kH, kW, ninput_chls, noutput_chls], stddev = init),
        name = 'weights')
    biases = tf.Variable(tf.zeros(noutput_chls), name='biases')

    conv = tf.nn.conv2d(inputs, weights, [1, dH, dW, 1], padding = padding)


def spatial_max_pooling_indices(inputs, kW, kH, dW=None, dH=None,
    padding='SAME', name=None):
    """Performs max pooling by taking the largest value in every kH x kW region,
    and outputs max values and pooling indices as tensors.

    Args:
        inputs: 4-D tensor with dimensions [batch_size, height, width, chls].
        kW: Width of region to pool over
        kH: Height of region to pool over
        dW: Horizontal stride (defaults to kW i.e. non-overlapping pooling)
        dH: Vertical stride (defaults to kH i.e. non-overlapping pooling)
        padding: A string value of either 'SAME' or 'VALID' (default 'SAME')
                'SAME': round down: w_out = floor((w_in - kW + 1)/dW)
                'VALID': round up: w_out = ceil((w_in - kW + 1)/dW)
        name: Name scope for the module

    Returns:
        output: 4-D feature map with dimensions [batch_size, h_output,
            w_output, chls]
        indices: 4D tensor which stores the flattened pooling indices"""

    if dW is None: dW = kW
    if dH is None: dH = kH
    return tf.nn.max_pool_with_argmax(inputs, [1, kH, kW, 1], [1, dW, dH, 1],
        padding, name)


def spatial_unpooling(inputs, indices, kW, kH, dW=None, dH=None, padding='SAME',
    name=None):
    """Reconstructs the input to a pooling operation using the saved pooling
    indices.

    Args:
        inputs: 4-D tensor with dimensions [batch_size, height, width, chls].
        indices: 4-D tensor with the same dimensions as inputs, containing the
            saved pooling indices
        kW: Width of pooling filter
        kH: Height of pooling filter
        dW: Horizontal stride (defaults to kW)
        dH: Vertical stride (defaults to kH)
        name: Name scope for the module

    Returns:
        output: Upsampled 4-D feature map with dimensions [batch_size,
            h_output, w_output, chls]"""
    if dW is None: dW = kW
    if dH is None: dH = kH

    # Compute output dimensions
    output_shape = (tf.shape(inputs) - [0, 1, 1, 0]) * [1, dH, dW, 1] + [0, kH, kW, 0]
    pnt = tf.Print(output_shape, [output_shape])

    # Flatten inputs and indices into 1D vectors
    inputs_flat = tf.reshape(inputs, [-1])
    indices_flat = tf.reshape(indices, [-1])
    output_shape_flat = tf.to_int64(tf.reduce_prod(output_shape, keep_dims = True))

    # Populate output with values from inputs
    output_flat = tf.sparse_to_dense(indices_flat, output_shape_flat, inputs_flat, 0)
    output = tf.reshape(output_flat, output_shape)

    return output
