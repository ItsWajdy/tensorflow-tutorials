from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28)/255.0


def create_model():
	model = keras.Sequential([
		keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(10, activation=tf.nn.softmax)
	])

	model.compile(optimizer=keras.optimizers.Adam(),
				  loss=keras.losses.sparse_categorical_crossentropy,
				  metrics=['accuracy'])

	return model


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Save the weights
model.save_weights('./checkpoints/my_checkpoints')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoints')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
