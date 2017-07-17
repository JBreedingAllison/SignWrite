"""
Build InceptionV3 based CNN to classify videos of sign language by training
on frames of videos in 'jpg' format in their labeled directories.
"""
# !/usr/bin/env python
import os
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from data_manager import DataSet

my_data = DataSet()

# Save models to checkpoints folder.
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

checkpointer = ModelCheckpoint(
    filepath='./checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Stop training when learning stops.
early_stopper = EarlyStopping(patience=10)

# Make logs for TensorBoard.
if not os.path.exists('./logs'):
    os.makedirs('./logs')

tensorboard = TensorBoard(log_dir='./logs/')


def get_generators():
    """
    Get the generators for the training and testing sets.
    output:
    train_generator, test_generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Set the batch_size and the accepted target_size for training/testing
    # for the InceptionV3 CNN.
    train_generator = train_datagen.flow_from_directory(
        './train/',
        target_size=(299, 299),
        batch_size=32,
        classes=my_data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        './test/',
        target_size=(299, 299),
        batch_size=32,
        classes=my_data.classes,
        class_mode='categorical')

    return train_generator, validation_generator


def get_model(weights='imagenet'):
    """
    Prepare the InceptionV3 based model for initial training.
    params:
    weights = the weights for the model, default to the ImageNet weights.
    output:
    model = our initial model
    """

    # Load the pre-trained model without the top layers.
    base_model = InceptionV3(weights=weights, include_top=False)

    # Add a global spatial average pooling layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer.
    x = Dense(1024, activation='relu')(x)

    # Add logistic layer.
    predictions = Dense(len(my_data.classes), activation='softmax')(x)

    # Define the model we will use for initial training.
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def get_top_layer_model(base_model):
    """
    Set the untrainable layers of a model
    params:
    model = the model we want to train
    output:
    model with layers set to untrainable.
    """
    # Set layers in base_model as untrainable.
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return base_model


def get_mid_layer_model(model):
    """
    Freeze the first 172 layers of the model and set the last layers
    to be trainable.
    params:
    model = the model
    output:
    the model with first 172 layers frozen and the last layers as trainable.
    """
    # Freeze the first 172 layers and let the final layers be trainable.
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # Recompile the model with SGD optimizer
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def train_model(model, num_epoch, generators, callbacks=[]):
    """
    Train the model.
    params:
    model = the model to train
    num_epoch = the number of training cycles
    generators = the training and tesing generators
    callbacks = the callbacks for training
    output:
    model = the retrained model
    """
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=num_epoch,
        callbacks=callbacks)
    return model


def main(weights):
    """
    Train the InceptionV3 based model.
    params:
    weights = the weights from the model. If none, use the pre-trained
              InceptionV3 ImageNet weights.
    """
    model = get_model()
    generators = get_generators()

    if weights is None:
        print("Loading the standard InceptionV3 model.")
        # Get and train the top layers.
        model = get_top_layer_model(model)
        model = train_model(model, 10, generators)
    else:
        print("Loading last version of the model: %s." % weights)
        model.load_weights(weights)

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    model = train_model(model, 100, generators,
                        [checkpointer, early_stopper, tensorboard])

if __name__ == '__main__':
    weights = None
    main(weights)
