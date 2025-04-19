# FalconEye: Military Aircraft Image Classifier (Full Training + Fine-Tuning)

FalconEye is a computer vision project that uses **Transfer Learning + Fine-Tuning** with TensorFlow and MobileNetV2 to classify **military aircraft** based on image data. Built using the **Military Aircraft Detection Dataset** from Kaggle (specifically the `crop` subfolder), this project represents a fusion of two passions: artificial intelligence and fighter jets.

---

## ✈️ Backstory

I’ve always been fascinated by the military—not war itself, but the sheer brilliance behind military aviation and machinery. The technology, engineering, and precision that go into these aircraft captivate me. Don’t get me wrong—I hate war. But man, I *love* jets.

There’s a solid 97% chance I can name any jet you show me, along with its nickname and variant. So for this project, I decided to bring together two things I love: **learning AI** and **identifying military jets**. How cool is that?

---

This project uses **Transfer Learning + Fine-Tuning** with the `MobileNetV2` architecture to classify **military aircraft** from the [Kaggle Military Aircraft Detection Dataset](https://www.kaggle.com/datasets).

✅ This README guides you through running the **entire workflow**, including:
- Data preparation and augmentation
- Initial training using transfer learning
- Fine-tuning for performance improvement
- Saving and visualizing model results

---

## 📂 Dataset Requirements

**Dataset Used:** Military Aircraft Detection Dataset from Kaggle  
📁 **This project specifically uses the `crop` subfolder** which this code splits into:
- `train`
- `validation`
- `test`
  
```python
train_dataset = tf.keras.utils.image_dataset_from_directory(PATH,validation_split = 0.3, shuffle = True, subset = "training", batch_size=
                 BATCH_SIZE, image_size= IMG_SIZE, seed= 123)
validation_dataset= tf.keras.utils.image_dataset_from_directory(PATH,validation_split = 0.3, shuffle = True, subset = "validation", batch_size
                 = BATCH_SIZE, image_size= IMG_SIZE, seed = 123)
```


🛑 **IMPORTANT**: Make sure your dataset directory is structured and passed correctly in the code:
```python
PATH = r"C:\Users\New\Documents\Military_dataset\crop"
```

## ✅ Run Instructions
Make sure all requirements are installed.
Download and extract the dataset.
Update all absolute file paths in the script:
Dataset path (PATH = ...)
Model save/load paths
Run the Python script (you may want to split into cells if using Jupyter Notebook).
```bash
python main.py
```

# 👩‍💻 Author
Kevin Kibet
Built for military aircraft recognition using transfer learning & fine-tuning.

## 🛠️ Current Status
I’m currently in the middle of training and fine-tuning, actively adjusting various hyperparameters—like the learning rate, number of layers unfrozen, and epochs—with the goal of reaching **94%+ validation accuracy**.
Once I hit that benchmark, I’ll upload the final saved model, its weights, and all relevant training metrics.
**BUT**—if you manage to hit or surpass that milestone before I do, feel free to reach out and let me know. I’d love to hear about your results and how you achieved them!