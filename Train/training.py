from time  import sleep
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import tensorflow as tf
import numpy as np

PATH = r"C:\Users\New\Path\To\Military_dataset\crop"

BATCH_SIZE= 25
IMG_SIZE = (160,160)
train_dataset = tf.keras.utils.image_dataset_from_directory(PATH,validation_split = 0.3, shuffle = True, subset = "training", batch_size= BATCH_SIZE, image_size= IMG_SIZE, seed= 123)
validation_dataset= tf.keras.utils.image_dataset_from_directory(PATH,validation_split = 0.3, shuffle = True, subset = "validation", batch_size = BATCH_SIZE, image_size= IMG_SIZE, seed = 123)

#Check and print some images from the dataset
#Create test datasets
validation_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(validation_batches // 5)
validation_dataset = validation_dataset.skip(validation_batches //5)

#configuring the dataset for performance by using buffered prefetching to load...
#...images without overloading or blocking I/O

AUTOTUNE = tf.data.AUTOTUNE
train_dataset =train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_batches = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#To introduce sample diversity, we augment the images i.e. by flipping,rotating e.t.c to avoid overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])

#Rescaling img to fit the [-1, 1] requirements of or base model(MobileNetV2)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#ceating the base model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights= 'imagenet')

#Freeze the base
base_model.trainable = False

#Feature extractor - converts 160 x 160 to 5 x 5 x 1280 block of features
image_batch , label_batch = next(iter(train_dataset))
feature_batch =base_model(image_batch)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(81, activation = 'softmax')#Adjust the size arg depending on the number of classes your dataset has, mine was 81
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(160,160,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

#run this to see the success of our model setup and its architecture:>> model.summary()

base_learning_rate = 0.0001
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= base_learning_rate), loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

initial_epochs = 15
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset, epochs = initial_epochs, validation_data = validation_dataset)

#print(history.history.keys()) to check if you got the right keys


sleep(10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


#At this point, we are finished with our transfer learning objective. So we save our model with what it has learned thus far.

model.save(r"C:\Users\New\Path\To\Initial_Trained\filename.keras")