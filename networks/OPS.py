import tensorflow as tf

def Conv_with_kernel(input, kernel, padding='SAME', name="conv"):
    with tf.variable_scope(name):
        return tf.nn.conv2d(input, filter=kernel, strides=(1, 1, 1, 1), padding=padding, use_cudnn_on_gpu=True,
                            data_format='NHWC', dilations=[1, 1, 1, 1], name=None)

def Conv_(input, filter, kernel=(3, 3), stride=(1, 1), padding='SAME', use_bias=False, name="conv"):
    with tf.variable_scope(name):
        return tf.layers.conv2d(
            input, filter, kernel, strides=stride, padding=padding,
            data_format='channels_last', dilation_rate=(1, 1), activation=None,
            use_bias=use_bias, kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None, trainable=True, name=None,
            reuse=None
        )

def Depth_conv_(input, filter, kernel=(3, 3), stride=(1, 1), padding='SAME', use_bias=False, name="conv"):
    with tf.variable_scope(name):
        return tf.layers.separable_conv2d(
            input, filter, kernel, strides=stride, padding=padding,
            data_format='channels_last', dilation_rate=(1, 1), depth_multiplier=1,
            activation=None, use_bias=use_bias, depthwise_initializer=None,
            pointwise_initializer=None, bias_initializer=tf.zeros_initializer(),
            depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None,
            bias_constraint=None, trainable=True, name=None, reuse=None
        )

def Maxpool_(input, kernel=(3, 3), stride=(2,2), padding='SAME', name="pool"):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(
            input, kernel, stride, padding=padding, data_format='channels_last',
            name=None
        )

def BN_(input, training, name="bn"):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(
            input, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(), beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
            training=training, trainable=True, name=None, reuse=None, renorm=False,
            renorm_clipping=None, renorm_momentum=0.99, fused=None, virtual_batch_size=None,
            adjustment=None
        )

def Relu_(x):
    return tf.nn.relu(x)

def LRelu_(x):
    return tf.nn.leaky_relu(x)

def Concat_(layers) :
    return tf.concat(layers, axis=3)

def Resize_(input, shape):
    return tf.image.resize_bilinear(input, (shape[0], shape[1]))

def Slice_(x, channel, slice_num):
    output_list = []
    single_channel = channel//slice_num
    for i in range(slice_num):
        out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        output_list.append(out)
    return output_list


def GAP(input):
    return tf.reduce_mean(input, axis=(1, 2), keepdims=True)

