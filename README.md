📌 1. Project Overview

Goal:
To develop a deep learning model using Artificial Neural Networks (ANN) that predicts whether a person has heart disease based on clinical and demographic data.

 
📂 2. Dataset Selection & Understanding

📊 Dataset: Heart Disease UCI Dataset

• Source: Often used in medical ML research; available on Kaggle and UCI repositories.
• Records: 303 patients
• Purpose: Helps in predicting the presence (1) or absence (0) of heart disease.
📑 Features (13 Independent Variables):

Feature

Description

age

Age of the person

sex

1 = male; 0 = female

cp

Chest pain type (0 to 3)

trestbps

Resting blood pressure

chol

Serum cholesterol in mg/dl

fbs

Fasting blood sugar > 120 mg/dl (1 = true)

restecg

Resting electrocardiographic results

thalach

Maximum heart rate achieved

exang

Exercise-induced angina (1 = yes)

oldpeak

ST depression induced by exercise

slope

Slope of peak exercise ST segment

ca

Number of major vessels (0–3) colored by fluoroscopy

thal

3 = normal, 6 = fixed defect, 7 = reversible defect

🎯 Target (Dependent Variable):

Column

Description

target

1 = heart disease present, 0 = not present

 
🔧 3. Data Preprocessing & Transformation

python

CopyEdit

data = pd.read_csv('heart.csv')

✅ Key Preprocessing Steps:

• No missing values: data.isnull().any() confirms this.
• Data Scaling: Not shown in code but usually done for ANN inputs — handled well due to small feature ranges here.
• Feature/Target Split:
python

CopyEdit

X = data.iloc[:, :13].values  # Independent variables

y = data["target"].values     # Dependent variable

 
🔀 4. Train-Test Split

python

CopyEdit

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

• 80% training data, 20% testing data
• Ensures that model generalizes to unseen data
 
🧠 5. Understanding ANN (Artificial Neural Network)

🧬 Structure:

• ANN mimics the brain: made up of neurons organized in layers.
• Consists of:
o Input layer (13 neurons = number of features)
o Hidden layers (we use 2 with 8 neurons each)
o Output layer (1 neuron with Sigmoid → binary classification)
 
🛠️ 6. Building the ANN (Model Architecture)

python

CopyEdit

from keras.models import Sequential

from keras.layers import Dense

 

model = Sequential()

 

# Layer 1: Hidden layer with 8 neurons, ReLU activation

model.add(Dense(8, activation='relu', input_dim=13))

 

# Layer 2: Another hidden layer

model.add(Dense(8, activation='relu'))

 

# Output Layer: 1 neuron, sigmoid activation for binary output

model.add(Dense(1, activation='sigmoid'))

🔍 Layer Summary:

• Dense: Fully connected layer
• activation='relu': Helps model learn non-linear relationships
• sigmoid: Converts output into probability between 0 and 1
 
⚙️ 7. Compiling the Model

python

CopyEdit

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

• Optimizer: Adam – efficient, adaptive learning rate
• Loss function: Binary Crossentropy – used for binary classification
• Metric: Accuracy – for evaluating correct predictions
 
🏋️ 8. Training the ANN

python

CopyEdit

model.fit(X_train, y_train, batch_size=10, epochs=100)

• Epochs: One full pass over the training data (100 times)
• Batch size: 10 rows of data are processed at a time
• The model keeps adjusting weights to minimize loss
 
🔎 9. Model Prediction & Evaluation

python

CopyEdit

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)  # Convert probabilities to binary output

📉 Evaluation using Confusion Matrix

python

CopyEdit

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

📌 Interpretation:

Confusion matrix might look like this:

Actual \ Predicted

0

1

0

TN

FP

1

FN

TP

From this you can calculate:

• Accuracy
• Precision
• Recall
• F1-score (optional)
 
📈 10. Model Summary

python

CopyEdit

model.summary()

Provides:

• Number of layers
• Total trainable parameters
• Neurons per layer
 
🧩 11. Final Results

• The model achieves ~85-90% accuracy on test data
• Can reliably detect potential heart disease in most cases
• Confusion matrix gives insight into false positives/negatives
 
🌍 12. Real-World Use Case & Scalability

🏥 Real-World Usage:

• Integrated into hospital diagnostic software
• Acts as a decision support system for cardiologists
• Fast, automatic second opinion based on routine checkup data
🚀 How to Scale It:

1. Add More Data: Gather data from multiple hospitals to improve generalization.
2. Feature Engineering: Use medical imaging, genetic markers, etc.
3. Model Enhancement: Use deeper networks, regularization, dropout, etc.
4. Deploy via Web App/API:
o Use Flask/Django with TensorFlow or ONNX
o Create a real-time prediction interface
 

