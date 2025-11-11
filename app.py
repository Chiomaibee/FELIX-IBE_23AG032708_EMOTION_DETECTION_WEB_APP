# app.py
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sqlite3
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create or connect to database
conn = sqlite3.connect('users.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    image_path TEXT,
                    emotion_result TEXT
                )''')
conn.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    file = request.files['file']
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Process image
    img = image.load_img(filepath, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]

    # Save to database
    cursor.execute("INSERT INTO users (name, image_path, emotion_result) VALUES (?, ?, ?)",
                   (name, filepath, emotion))
    conn.commit()

    return render_template('index.html', prediction=emotion, image=filepath)

if __name__ == '__main__':
    app.run(debug=True)
