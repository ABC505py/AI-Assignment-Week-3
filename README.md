Part 1: Theoretical Understanding (40%)
Q1: Differences between TensorFlow and PyTorch

| Feature               | TensorFlow                          | PyTorch                               |
| --------------------- | ----------------------------------- | ------------------------------------- |
| **Computation Graph** | Static (Graphs defined before run)  | Dynamic (Graphs built at runtime)     |
| **Ease of Debugging** | More difficult due to static graphs | Easier with Python-native debugging   |
| **Deployment**        | TensorFlow Serving, TensorFlow Lite | TorchScript, ONNX (limited ecosystem) |
| **Preferred Use**     | Production-ready deployment         | Research, experimentation             |


When to choose:

TensorFlow: For robust deployment and mobile inference.

PyTorch: For rapid prototyping and research.

Q2: Use Cases for Jupyter Notebooks in AI Development
Prototyping & Experimentation: Easily test and modify ML models in real-time with visual outputs.

Documentation & Sharing: Combine code, results, and explanations in one document ideal for collaboration and teaching.

Q3: How spaCy Enhances NLP
Advanced Linguistic Features: spaCy provides tokenization, POS tagging, and NER.

Speed & Accuracy: Optimized Cython backend processes large text quickly.

Compared to Basic String Ops: spaCy understands context (e.g., "Apple" as a brand vs. fruit), whereas string ops treat text as raw characters.

| Feature                 | Scikit-learn                             | TensorFlow                               |
| ----------------------- | ---------------------------------------- | ---------------------------------------- |
| **Target Applications** | Classical ML (SVM, k-NN, decision trees) | Deep Learning (CNNs, RNNs, Transformers) |
| **Ease of Use**         | Very beginner-friendly                   | Steeper learning curve                   |
| **Community Support**   | Strong for ML beginners                  | Strong for advanced DL users             |


Part 2: Practical Implementation (50%)
Task 1: Decision Tree Classifier (Scikit-learn)
python
Copy code
# iris_classifier.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
Task 2: CNN on MNIST (TensorFlow)
python
Copy code
# mnist_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train[..., None]/255.0, x_test[..., None]/255.0

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Visualize predictions
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()}, True: {y_test[i]}")
    plt.show()
Task 3: NLP with spaCy
python
Copy code
# nlp_spacy.py
import spacy

nlp = spacy.load("en_core_web_sm")

review = "I love the sound quality of Bose headphones. The Sony ones are also decent."

# NER
doc = nlp(review)
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Rule-based Sentiment
if "love" in review.lower() or "great" in review.lower():
    print("Sentiment: Positive 😊")
elif "hate" in review.lower() or "bad" in review.lower():
    print("Sentiment: Negative 😠")
else:
    print("Sentiment: Neutral 😐")
Part 3: Ethics & Optimization (10%)
1. Ethical Considerations
Bias Risks:

MNIST may not represent all handwriting styles.

Amazon reviews may skew toward English or popular brands.

Solutions:

Use TensorFlow Fairness Indicators to measure bias by group (e.g., digits, authors).

Use spaCy's rule-based sentiment with human review to reduce annotation bias.

2. Troubleshooting Challenge (Sample Bug Fix)
Bug:

python
Copy code
# Incorrect loss for classification
model.compile(optimizer='adam', loss='mean_squared_error')
Fix:

python
Copy code
# Correct classification loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
Bug:

python
Copy code
# Incorrect shape
model.fit(x_train, y_train.reshape(-1, 1), ...)
Fix:

python
Copy code
# Keep y_train as (batch_size,)
model.fit(x_train, y_train, ...)
