from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import ZeroPadding3D, Lambda
import tensorflow as tf


def contract2d(prev_layer, n_kernel, kernel_size, pool_size, padding, act):
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    pool = MaxPooling2D(pool_size=pool_size, strides=pool_size)((conv))
    return conv, pool


def expand2d(prev_layer,
             left_layer,
             n_kernel,
             kernel_size,
             pool_size,
             padding,
             act,
             dropout=False):
    up = Concatenate(axis=-1)(
        [UpSampling2D(size=pool_size)(prev_layer), left_layer])
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(up)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel >> 1
    if dropout:
        conv = Dropout(.25)(conv)
    conv = Conv2D(n_kernel, kernel_size, padding=padding)(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    return conv


def contract3d(prev_layer, n_kernel, kernel_size, pool_size, padding, act,
               pooling):
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(prev_layer)
    conv = Activation(act)(BatchNormalization()(conv))
    n_kernel = n_kernel << 1
    conv = Conv3D(n_kernel, kernel_size, padding='valid')(conv)
    conv = Activation(act)(BatchNormalization()(conv))
    conv = ZeroPadding3D(padding=(0, 2 * (kernel_size // 2),
                                  2 * (kernel_size // 2)))(conv)
    if pooling:
        pool = MaxPooling3D(pool_size=pool_size, strides=pool_size)((conv))
        return conv, pool
    else:
        return conv, None


def build_model(input_shape, output_ch=1, dropout=True, softmax_output=True):
    inputs = Input(input_shape)

    kernel_size = 3
    pool_size = 2
    padding = 'same'
    activation = 'relu'
    n_kernel = 16

    pool = Lambda(lambda x: tf.expand_dims(x, axis=-1))(
        inputs)  # add channel axis
    enc3ds = []
    for _ in range(1):
        conv, pool = contract3d(pool,
                                n_kernel,
                                kernel_size,
                                pool_size,
                                padding,
                                activation,
                                pooling=True)
        enc3ds.append(conv)
        n_kernel = conv.shape[-1]

    pool = Lambda(lambda x: tf.squeeze(x, 1))(pool)
    encs = []
    for _ in range(5):
        conv, pool = contract2d(pool, n_kernel, kernel_size, pool_size,
                                padding, activation)
        encs.append(conv)
        n_kernel = conv.shape[-1]

    for i, enc in enumerate(encs[-2::-1]):
        conv = expand2d(conv,
                        enc,
                        n_kernel,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        dropout=dropout)
        n_kernel = conv.shape[-1]

    for i, enc in enumerate(enc3ds[-1::-1]):
        enc = Conv3D(n_kernel,
                     (enc.shape[1], 1, 1))(enc)  # reduce along z axis
        enc = Lambda(lambda x: tf.squeeze(x, 1))(enc)
        conv = expand2d(conv,
                        enc,
                        n_kernel,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        dropout=dropout)
        n_kernel = conv.shape[-1]

    if output_ch > 1:
        if softmax_output:
            output = Conv2D(output_ch,
                            1,
                            padding=padding,
                            activation='softmax')(conv)
        else:
            output = Conv2D(output_ch, 1, padding=padding,
                            activation='linear')(conv)
    else:
        output = Conv2D(output_ch, 1, padding=padding,
                        activation='sigmoid')(conv)

    return Model(inputs=inputs, outputs=output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Slab U-Net model.')
    parser.add_argument('output',
                        help='Output image filename',
                        metavar='<output>',
                        nargs='?')

    args = parser.parse_args()

    model = build_model((6, 512, 512), 5)
    if args.output:
        from tensorflow.keras.utils import plot_model
        plot_model(model,
                   to_file=args.output,
                   show_shapes=True,
                   show_layer_names=False)
    model.summary()


if __name__ == '__main__':
    import sys
    sys.exit(main())
