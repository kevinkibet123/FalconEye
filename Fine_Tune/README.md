# Part 2: Fine Tuning
### NOTE: 
Make sure that you load the same ***.keras*** file that was created and trained in part 1 for this step. Otherwise you risk losing everything the model had learned.

## 📌 Objective
Enhance model performance by unfreezing some layers of the pre-trained MobileNetV2 base model and retraining with a lower learning rate.

## 🔄 Workflow
1. Load the model trained in Part 1.
2. Unfreeze MobileNetV2 layers from index 100 onwards.
3. Recompile the model with:
   - Optimizer: `RMSprop`
   - Learning Rate: `base_learning_rate / 10`
4. Continue training for 12 additional epochs.

## ⚙️ Fine-Tuning Parameters
- Fine-tuning Layers: Last 50+ layers
- Total Epochs: 27 (`15 + 12`)
- Optimizer: RMSprop
  
## 📝 Notes
- Ensures better generalization.
- Helps learn dataset-specific features.
- Saves computation by only unfreezing deeper layers.

## 🚀 How to Run
```bash
python fine_tune.py
```
## ✅ Output
Fine-tuned model saved as:
```bash
C:\Users\New\Path\To\initial_trained_model.keras
```

# Optional features
You can run fine-tuning multiple times as needed. Just make sure to:
- Adjust fine_tune_at to unfreeze different layers.
- Modify learning_rate, epochs, and optionally apply regularization or learning rate decay.

### 🚨 EarlyStopping Callback (Highly Recommended)
To save time and computational resources, especially if fine-tuning in multiple rounds over long training cycles, use the EarlyStopping callback.
```python
early_stopping = tf.keras.callbacks.EarlyStopping( 
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
```
Include the callback in your model.fit() function:
```python
model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
```
*🧠 patience=5 means training will stop if validation accuracy doesn't improve for 5 consecutive epochs. This helps prevent overfitting and reduces unnecessary training time.*

## 📝 Notes
- Fine-tuning can be repeated multiple times with adjusted settings.
- Safe layer unfreezing typically starts from layer 90+.
- Use model.summary() to view and plan which layers to unfreeze.
- Don't forget to reduce the learning rate during fine-tuning.
- EarlyStopping is a great addition when experimenting with:
- - Longer fine-tuning epochs
- - Small validation improvements
- - Time/resource constraints