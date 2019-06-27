'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
from data_utils import SortedNumberGenerator, SumNumberGenerator, SkipNumberGenerator
import os
import keras
from keras import backend as K
import tensorflow as tf

from training_utils import SaveEncoder, res_block, make_periodic_lr

def cross_entropy_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, dim=2)
    return tf.reduce_mean(loss)

def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x

def network_encoder_resnet(x, code_size):
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='relu')(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)
    return x


def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)

    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    # x = keras.layers.BatchNormalization()(x)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = preds[:, :, None, :] * y_encoded # this should be broadcasted to N x T_pred x (negative_samples + 1) x code_size
        ret = K.sum(dot_product, axis=-1)

        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[1][:3]


def network_cpc(image_shape, terms, predict_terms, negative_samples, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder_resnet(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)
    # If we want to ditch the RNN and stack the history instead, uncomment the following two lines of code
    # context = keras.layers.Reshape((code_size * terms,))(x_encoded)
    # context = keras.layers.Dense(512, activation='relu')(context)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, (negative_samples + 1), image_shape[0], image_shape[1], image_shape[2]))
    y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), *image_shape))(y_input)
    y_encoded_flat = keras.layers.TimeDistributed(encoder_model)(y_input_flat)
    y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), code_size))(y_encoded_flat)

    # Loss
    logits = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=logits)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=cross_entropy_loss,
        metrics=['categorical_accuracy']
    )
    cpc_model.summary()

    return cpc_model


def train_model(epochs, batch_size, output_dir, code_size, task='sorted', lr=1e-4, terms=4, predict_terms=4, negative_samples=5, max_skip_step=2, image_size=28, color=False):

    # Prepare data
    if task == 'sorted':
        train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                           negative_samples=negative_samples, predict_terms=predict_terms,
                                           image_size=image_size, color=color, rescale=True)
        validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                                negative_samples=negative_samples, predict_terms=predict_terms,
                                                image_size=image_size, color=color, rescale=True)
    elif task == 'sum':
        steps = list(range(1, max_skip_step + 1))
        train_data = SumNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                           negative_samples=negative_samples, steps=steps,
                                           image_size=image_size, color=color, rescale=True)
        validation_data = SumNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                                negative_samples=negative_samples, steps=steps,
                                                image_size=image_size, color=color, rescale=True)
    elif task == 'skip':
        steps = list(range(1, max_skip_step + 1))
        train_data = SkipNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                           negative_samples=negative_samples, predict_terms=predict_terms, steps=steps,
                                           image_size=image_size, color=color, rescale=True)
        validation_data = SkipNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                                negative_samples=negative_samples, predict_terms=predict_terms, steps=steps,
                                                image_size=image_size, color=color, rescale=True)
    else:
        raise NotImplementedError("Taks %s is not supported!" % task)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepares the model
    model = network_cpc(image_shape=(image_size, image_size, 3), terms=terms, predict_terms=predict_terms,
                        negative_samples=negative_samples, code_size=code_size, learning_rate=lr)

    # Callbacks
    callbacks = [#keras.callbacks.LearningRateScheduler(make_periodic_lr([5e-2, 5e-3, 5e-4]), verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=4, min_lr=1e-5, verbose=1, min_delta=0.001),
                 SaveEncoder(output_dir)]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    #
    # model.save(os.path.join(output_dir, 'cpc.h5'))
    #
    # # Saves the encoder alone
    # encoder = model.layers[1].layer
    # encoder.save(os.path.join(output_dir, 'encoder.h5'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train CPC model')
    parser.add_argument('task', type=str, help='the task CPC tries to complete')
    parser.add_argument('terms', type=int, help='number of history terms taken into account')
    parser.add_argument('predict_terms', type=int, help='number of future terms to predict')
    parser.add_argument('negative_samples', type=int, help='number of negative samples')
    parser.add_argument('--max_skip_step', type=int, default=2, help='maximum step size for skip task')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train the model for')
    parser.add_argument('--run_suffix', type=str, default='')

    args = parser.parse_args()

    output_dir = 'models/64x64'
    exp_name = '%s_hist%d_fut%d_neg%d' % (args.task, args.terms, args.predict_terms, args.negative_samples)
    if args.task == 'skip' or args.task == 'sum':
        exp_name += '_skip%d' % args.max_skip_step + '_%s' % args.run_suffix
    output_dir = os.path.join(output_dir, exp_name)

    train_model(
        epochs=args.epoch,
        batch_size=32,
        output_dir=output_dir,
        code_size=128,
        task=args.task,
        lr=1e-3,
        terms=args.terms,
        predict_terms=args.predict_terms,
        negative_samples=args.negative_samples,
        max_skip_step=args.max_skip_step,
        image_size=64,
        color=True
    )

