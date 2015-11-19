import tensorflow as tf

def spatial_conv(input, noutput_chls, kW, kH, dW=1, dH=1,
    padding = "SAME", init = 1.0, name=None):
    """Convolutional layer which applies a 2D filter over an image.

    Args:
        input: 4-D tensor with dimensions [batch_size, height,
            width, ninputchls].
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

    ninputchls = input.get_shape().as_list()[3]

    weights = tf.Variable(
        tf.truncated_normal([kH, kW, ninputchls, noutput_chls], stddev = init),
        name = 'weights')
    biases = tf.Variable(tf.zeros(noutput_chls), name='biases')

    conv = tf.nn.conv2d(input, weights, [1, dH, dW, 1], padding = padding)


def spatial_max_pooling_indices(input, kW, kH, dW=None, dH=None,
    padding='SAME', name=None):
    """Performs max pooling by taking the largest value in every kH x kW region,
    and outputs max values and pooling indices as tensors.

    Args:
        input: 4-D tensor with dimensions [batch_size, height, width, chls].
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
    return tf.nn.max_pool_with_argmax(input, [1, kH, kW, 1], [1, dW, dH, 1],
        padding, name)


def spatial_unpooling(input, indices, kW, kH, dW=None, dH=None,
    output_shape=None, name=None):
    """Reconstructs the input to a pooling operation using the saved pooling
    indices.

    Args:
        input: 4-D tensor with dimensions [batch_size, height, width, chls].
        indices: 4-D tensor with the same dimensions as input, containing the
            saved pooling indices
        kW: Width of pooling filter
        kH: Height of pooling filter
        dW: Horizontal stride (defaults to kW)
        dH: Vertical stride (defaults to kH)
        output_shape: Desired output shape. If ommitted, the output shape is
            inferred from the input dimensions and kernel size
        name: Name scope for the module

    Returns:
        output: Upsampled 4-D feature map with dimensions [batch_size,
            h_output, w_output, chls]"""
    if dW is None: dW = kW
    if dH is None: dH = kH

    # Compute output dimensions
    if output_shape is None:
        output_shape = ((tf.shape(input) - [0, 1, 1, 0]) * [1, dH, dW, 1]
            + [0, kH, kW, 0])

    # Flatten input and indices into 1D vectors
    input_flat = tf.reshape(input, [-1])
    indices_flat = tf.reshape(indices, [-1])
    output_shape_flat = tf.to_int64(tf.reduce_prod(output_shape,
        keep_dims = True))

    # Populate output with values from input
    output_flat = tf.sparse_to_dense(indices_flat, output_shape_flat,
        input_flat, 0)
    output = tf.reshape(output_flat, output_shape)

    return output


def spatial_batch_normalization(input, eps=1e-5, name=None):
    """Performs batch normalization as described in "Batch Normalization:
    Accelerating Deep Network Training by Reducing Internal Covariate Shift",
    Ioffe, Szegedy, 2015

    Args:
        input: 4-D tensor with dimensions [batch_size, height, width, chls]
        eps: A small scalar added to the variance to ensure numerical stability
            (default 1e-5)
        name: Variable scope for the module

    Returns:
        output: A batch normalized tensor with the same dimensions as the input

    """

    # Compute mean and variance over the batch
    mean, var = tf.nn.moments(input, [0])

    # Reshape mean and variance tensors
    input_shape = tf.shape(input).as_list()

    # Define affine transformation parameters
    beta = tf.Variable(tf.truncated_normal(input_shape[1:4]), name="beta")
    gamma = tf.Variable(tf.truncated_normal(input_shape[1:4]), name="gamma")

    # Combine gamma and variance into a single scaling term for efficiency
    scale = tf.div(gamma, tf.sqrt(var + eps))

    # Replicate tensors along batch dimension
    mean_t = tf.tile(tf.expand_dims(mean, 0), [input_shape[0], 1, 1, 1])
    beta_t = tf.tile(tf.expand_dims(beta, 0), [input_shape[0], 1, 1, 1])
    scale_t = tf.tile(tf.expand_dims(scale, 0), [input_shape[0], 1, 1, 1])

    # Apply normalization and affine transform
    output = (input - mean_t) * scale_t + beta_t

    return output
