import matplotlib.pyplot as plt # awesome library for vizualising data, good thing i discover this
import numpy as np
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist


(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#normalize the data 
training_images = training_images / 255.0
test_images = test_images / 255.0


# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])






model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(training_images, training_labels, epochs = 5)

model.evaluate(test_images, test_labels)

#exercise 1 

classifications = model.predict(test_images)


print(classifications[0])

print(test_labels[0])
