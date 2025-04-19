# Part 3: Monitoring & Evaluation
This section handles **monitoring model performance** and **testing** our trained and fine-tuned MobileNetV2-based model on unseen data. It includes:
- Visualizing training/validation metrics
- Generating predictions on test data
- Displaying results with matplotlib

---

## 📌 What’s Included

### 1. **Metric Tracking and Visualization**
Plots model accuracy and loss over time—both for the initial transfer learning phase and fine-tuning.

> 🔁 You can **append the monitoring code directly to your `fine_tune.py` script** for simplicity.  
> Alternatively, you can **import** the following variables:
> - `history`, `acc`, `val_acc`, `loss`, `val_loss` from `training.py`
> - `history_fine` from `fine_tune.py`

But again, it's easier to just append the code to `fine_tune.py` and run everything together at once.

---

### 2. **Model Testing Code**
The appended testing script does the following:
- Loads your trained model
- Uses the `test_dataset` extracted from the original validation split
- Makes predictions
- Compares them against actual labels
- Visualizes the predictions on random test images

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

PATH = r"C:\Users\New\Documents\Military_dataset\crop"

BATCH_SIZE = 25
IMG_SIZE = (160,160)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    PATH, validation_split=0.3, shuffle=True, subset="training",
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=123)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    PATH, validation_split=0.3, shuffle=True, subset="validation",
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=123)

validation_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(validation_batches // 5)
validation_dataset = validation_dataset.skip(validation_batches // 5)

class_names = train_dataset.class_names

# Load the fine-tuned model
model = tf.keras.models.load_model(
    r"C:\Users\New\PycharmProjects\AI_Falconer\AI_FalconEye_2.keras")

# Run predictions on a test batch
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)



# Visualize the predictions
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()

```


## 📊 Visualizations
- Training Accuracy vs. Epoch
- Validation Accuracy vs. Epoch
- Training Loss vs. Epoch
- Validation Loss vs. Epoch

## 🖼️ Plot Features
- Mark where fine-tuning starts
- Shows overfitting if any
- Track training improvements

## 🚀 How to Run
```bash
python monitor.py
```

## 📁 Output
- Interactive plot displayed using matplotlib
Optional: Save as .png using: 
```python
plt.savefig("plot.png")
```
## ✅ What You’ll Get
- A visual summary of 9 randomly selected test predictions.
- Printed prediction vs actual label output.
- A quick way to manually verify classification quality.