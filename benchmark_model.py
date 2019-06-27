''' This module evaluates the performance of a trained CPC encoder '''

from data_utils import MnistGenerator
import keras
import os


def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, image_size=28, color=False):

    # Prepare data
    train_data = MnistGenerator(batch_size, subset='train', image_size=image_size, color=color, rescale=True)

    validation_data = MnistGenerator(batch_size, subset='valid', image_size=image_size, color=color, rescale=True)

    # Prepares the model
    model = build_model(encoder_path, image_shape=(image_size, image_size, 3), learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-5, verbose=1, min_delta=0.01)]

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
    model.save(os.path.join(output_dir, 'supervised.h5'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CPC model')
    parser.add_argument('task', type=str, help='the task CPC tries to complete')
    parser.add_argument('terms', type=int, help='number of history terms taken into account')
    parser.add_argument('predict_terms', type=int, help='number of future terms to predict')
    parser.add_argument('negative_samples', type=int, help='number of negative samples')
    parser.add_argument('--max_skip_step', type=int, default=2, help='maximum step size for skip task')
    parser.add_argument('--run_suffix', type=str, default='')

    args = parser.parse_args()

    output_dir = 'models/64x64'
    exp_name = '%s_hist%d_fut%d_neg%d' % (args.task, args.terms, args.predict_terms, args.negative_samples)
    if args.task == 'skip' or args.task == 'sum':
        exp_name += '_skip%d' % args.max_skip_step + '_%s' % args.run_suffix
    output_dir = os.path.join(output_dir, exp_name)

    benchmark_model(
        encoder_path=os.path.join(output_dir, 'encoder.h5'),
        epochs=15,
        batch_size=64,
        output_dir=output_dir,
        lr=1e-3,
        image_size=64,
        color=True
    )
