import os
import shutil
import numpy as np
import tensorflow as tf
import joblib
import asyncio

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from datetime import datetime
from gevent.pywsgi import WSGIServer

import firebase_admin
from firebase_admin import credentials, storage

load_dotenv()
cred = credentials.Certificate('credentials/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'baskaryaapp.appspot.com'})

bucket = storage.bucket()

app = Flask(__name__)

# Load your custom motif batik classification model
motif_batik_model = load_model('model.h5')

# Use the layer before softmax as feature extractor
# feature_extractor = motif_batik_model.get_layer('feature_extract_layer')
feature_extractor = tf.keras.Model(inputs=motif_batik_model.input, outputs=motif_batik_model.get_layer('feature_extract_layer').output)

recomendation_path = 'static/rekomendasi'

def getFileName(path):
  path = path.split('/')[3]
  path = path.replace('-', ' ')
  words = [word.capitalize() for word in path.split()[:-1]]
  result = ' '.join(words)

  return result

def getFileUrl(path):
  path_split = path.split('/')
  url = f'https://firebasestorage.googleapis.com/v0/b/baskaryaapp.appspot.com/o/batikAssets%2F{path_split[2]}%2F{path_split[3]}?alt=media&token=88c0220f-1239-49a5-988c-5e0cb54557da'

  return url

def load_and_process_image(image_path):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

imported_images = []

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Memuat fitur gambar dan list path gambar
images_features = joblib.load('image_features.joblib')
image_paths = joblib.load('image_paths.joblib')

# API FOR MACHINE LEARNING
@app.route('/api/ml', methods=['GET', 'POST'])
def index():
    result_images = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '' or not allowed_file(uploaded_file.filename):
            return jsonify({'status': 'error', 'error' : True, 'message': 'Invalid file. Please upload a valid image file.'}), 403

        # Save the uploaded file
        uploaded_file.save('static/uploaded_image.jpg')

        # Process the uploaded image
        uploaded_image = load_and_process_image('static/uploaded_image.jpg')

        # Extract features using the classification model
        uploaded_features = feature_extractor.predict(uploaded_image)

        # Compute cosine similarities
        similarities = cosine_similarity(uploaded_features, images_features)

        # Get the most similar images
        target_image_index = 0
        recommended_images = np.argsort(similarities[target_image_index])[::-1][:5]
        result_images = [(recomendation_path + '/' + image_paths[i])for i in recommended_images]

        output = []
        for image_path in result_images:
          file_name = getFileName(image_path)
          file_url = getFileUrl(image_path)
          
          entry = {
              'Nama Batik': file_name,
              'Url': file_url
          }
          
          output.append(entry)

        return jsonify({'status': 'success', 'error' : False, 'similar_images': output}), 201

if __name__ == '__main__':
    app.run(
      debug = True
    )
