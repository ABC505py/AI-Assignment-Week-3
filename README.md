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
