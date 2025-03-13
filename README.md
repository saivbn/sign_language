Sign Language Recognition Using CNN

📚 Project Overview
This project aims to build a sign Language Recognition System using Convolutional Neural Network (CNN)
with tensorFlow and Keras. The model is trained to classify different sign language gestures and predict them accurately

🛠️ Technologies Used
1. Python 3.12
2. TensorFlow 2.x
3. Keras
4. Opeancv
5. Numpy
6. Mediapipe
7. Pandas

📂 Project Structure

   sign_language/
│
├── sign_language_data/
│   ├── dataset/
│   ├── CNN.py
│   ├── sign_language_model.h5
│   └── sign_language_model.keras
│
├── models/
├── utils/
└── README.md

🖥️ Installation Guide


Step 1: Clone the Repository
   git clone https://github.com/saivbn/sign_language.git
cd sign_language


step 2: Install Dependencies
pip install tensorFloe opencv-python Mediapipe numpy

🚀 Training the Model


python sign_language_data/CNN.py


📈 Model Training Results


1. Training Accuracy:93.51%
2. Validation Accuracy:91.88%

📤 Saving the Model


model.save("sign_language_model.keras")

🧐 Testing the Model


import tensorflow as tf
model = tf.keras.models.load_model("sign_language_model.keras")

✅ Future Improvements


1. Add more sign language datasets.
2. Implement real-time gesture recognition with openCV
3. Deploy the model as a web app

📧 Contact


GitHub    - saivbn 
Email     - saik62907@gmail.con
Instagram - sai__2_7
Phone no. - 7286813844
