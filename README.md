# Breast Ultrasound Image Classification

The main goal of this project is to **classify breast ultrasound images** as **benign (non-cancerous)** or **malignant (cancerous)** based on their visual features.  

Although the dataset is relatively simple and contains only **250 images**, it serves as a good example to demonstrate my understanding of:  
- Data preprocessing  
- Handling class imbalance  
- Building classification models  

---

# Data Preparation and Preprocessing

In this project, I:  

- Loaded breast ultrasound images from **benign** and **malignant** folders  
- Converted images to **grayscale** and resized them to **28×28 pixels**  
- Normalized and flattened the image arrays  
- Split the dataset into **training** and **testing** sets  
- Applied **SMOTE** to handle class imbalance and create a balanced training set  

---

# Model Building and Evaluation

- Trained and tested three machine learning algorithms:  
  - **SVM** (Support Vector Machine)  
  - **Random Forest**  
  - **KNN** (K-Nearest Neighbors)  
- Since this was a practice dataset, the models achieved **very high accuracy** (close to 100%).  

---

# Purpose

The main purpose of this project was to practice the **full workflow**:  
- From image preprocessing  
- To handling class imbalance  
- To training and evaluating machine learning models  

All using a **clear and manageable dataset**, which makes it ideal for learning and demonstrating skills.  
