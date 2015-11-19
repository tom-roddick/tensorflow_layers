import tensorflow as tf

def spatial_cross_entropy(logits, labels, weights=None, name=None):
    """Calculates the pixelwise cross entropy loss: similar to
    tf.softmax_cross_entropy_with_logits but instead accepts a 4D tensor.

    Args:
        logits: Un-normalized probability distribution over classes for each
            spatial location. 4-D tensor with dims [batch_size, height,
            width, n_classes]. logits are expected to be unscaled so
            spatial_cross entropy should not be used after a softmax layer.
        labels: Accepts either a 3-D tensor with dimensions [batch_size, height,
            width], where each element represents a class index; or a 4-D tensor
            of dimensions [batch_size, height, width, n_classes], where the last
            dimension is a one-hot vector for the n_classes
        weights: Optionally weight each class by the given weighting factors
        name: Name scope for the module

    Returns:
        loss: The total cross-entropy loss"""

    # Determine dimensions
    input_shape = tf.shape(logits)
    batch = input_shape[0] * input_shape[1] * input_shape[2]
    n_classes = input_shape[3]
    output_shape = tf.pack([batch, n_classes])

    # Flatten inputs
    logits_flat = tf.reshape(logits, output_shape)
    labels_flat = tf.expand_dims(tf.reshape(labels, tf.pack([batch])),1)

    # Convert labels to one-hot vectors
    if labels.get_shape().ndims == 3:
        indices = tf.expand_dims(tf.range(0, batch, 1), 1)
        label_inds = tf.concat(1, [indices, labels_flat])
        labels_onehot = tf.sparse_to_dense(label_inds, output_shape,
            1.0, 0.0)
    else:
        labels_onehot = labels

    # Apply weights
    if weights is not None:
        weights_t = tf.tile(tf.expand_dims(weights, 0), [batch, 1])
        labels_w = tf.mul(labels_onehot, weights_t)
    else:
        labels_w = labels_onehot

    # Evaluate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_flat, labels_w)
    loss = tf.reduce_sum(cross_entropy)

    return loss
