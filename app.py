import numpy as np
import pandas as pd
import os
from PIL import Image
#load,clean and prepare the data
mainpath = "/data/Breast Ultrasound Image"
categories = ['benign', 'malignant']
image_size=(28,28)


x=[]
y=[]
for lable , category in enumerate(categories):
    folder_path=os.path.join(mainpath,category)
    for filename in os.listdir(folder_path):
        file_path=os.path.join(folder_path,filename)
        try:
            img=Image.open(file_path).convert('L')
            img=img.resize(image_size)
            img_array=np.array(img)
            x.append(img_array)
            y.append(lable)
        except Exception as e:
            print(f"Could not process {file_path}: {e}")
x=np.array(x)
y=np.array(y)


print(x)
x=x/255.0
x_flat = x.reshape(len(x), -1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42)


unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Label {label}: {count} samples")


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x_train, y_train)

print("Original dataset shape is:", x_train.shape, y_train.shape)
print("Resampled dataset shape is:", X_resampled.shape, y_resampled.shape)
unique, counts = np.unique(y_resampled, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Label {label}: {count} samples")


#SVM model

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_resampled, y_resampled)
y_pred = svm_model.predict(x_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Random Forest model

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
y_pred_rf = rf_model.predict(x_test)
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


#KNN model

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_resampled, y_resampled)
y_pred_knn = knn_model.predict(x_test)
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))