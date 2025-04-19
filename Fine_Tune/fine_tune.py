import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf


#load trained model from where we first saved it.
model = tf.keras.models.load_model(r"C:\Users\New\Path\To\Initial_Trained\filename.keras")

base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =False

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate/10),loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

fine_tune_epochs = 12
total_epochs = fine_tune_epochs + initial_epochs

history_fine= model.fit(train_dataset, epochs = total_epochs, initial_epoch= len(history.epoch), validation_data = validation_dataset)

model.save(r"C:\Users\New\Path\To\Fine_Tuned_Model\filename1.keras")