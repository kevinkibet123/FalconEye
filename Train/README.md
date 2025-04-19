# Part 1: Model Training

## 📌 Objective
Train an image classification model using **Transfer Learning** with MobileNetV2 on a military aircraft dataset. This phase prepares a robust feature extractor by freezing the base model and training the top layers on the custom dataset.

## 📁 Dataset
***Dataset obtained from Military Aircraft Detection Dataset [https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/code] by T. Nakamura on Kaggle***
### 📁 Dataset Requirements
- Your dataset should be organized in subfolders by class under a common directory.
- Update the `PATH` variable in the script to point to your local dataset:
  ```python
  PATH = r"C:\Absolute\Path\To\Your\Image\Dataset"
- Format: Folder-based class structure (one folder per class)
- Total Classes: 81 (can be auto-detected) by appending:
    '''python
    num_classes = len(train_dataset.class_names)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')


## 🧠 Model Architecture
- Base: MobileNetV2 (pre-trained on ImageNet)
- Layers:
  - `RandomFlip`, `RandomRotation` (Data Augmentation)
  - `GlobalAveragePooling2D`
  - `Dropout`
  - `Dense(81, activation='softmax')` (classifier layer)

## ⚙️ Key Parameters
- Image Size: 160 x 160
- Batch Size: 25
- Validation Split: 30%
- Base Learning Rate: 0.0001
- Epochs: 15
- Number of Classes: 81 (change prediction_layer = Dense(num_classes) as needed)

## 🧩 Training Workflow Overview
Image Loading:
- Uses image_dataset_from_directory with a training/validation split.

Dataset Performance Optimization:
- Applies buffered prefetching with tf.data.AUTOTUNE to prevent I/O bottlenecks.

Data Augmentation:
- Applies random flips and rotations to increase data diversity and prevent overfitting.

Preprocessing:
- Uses MobileNetV2’s preprocess_input for normalization into the [-1, 1] range.

Model Definition:
- Freezes base model to keep ImageNet weights.
- Builds classification head on top.

Evaluation:
- Evaluates model performance before training.

Training:
- Runs for 15 epochs on the prepared dataset.

Saving:
- Saves the model in .keras format. Update this path in your script:
    ```python
    model.save(r"C:\Users\New\Path\To\filename.keras")
    ```

## 🚀 How to Run
First edit the save location in model.save() by pasing the absolute path reference as the argument.

```bash
python training.py
