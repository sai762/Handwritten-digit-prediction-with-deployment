from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
model_mlp = tf.keras.models.load_model('mlp_baseline_model.h5')
model_cnn = tf.keras.models.load_model('cnn_model_with_augmentation.h5')
model_lenet = tf.keras.models.load_model('mnist_lenet_model.h5')

# Define a function to preprocess the image for MLP model
def preprocess_image_mlp(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.flatten()
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to preprocess the image for CNN and LeNet models
def preprocess_image_cnn(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

# Define a route for the homepage
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'model' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        model_choice = request.form['model']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Open the image
            image = Image.open(io.BytesIO(file.read()))
            if model_choice == 'mlp':
                image_array = preprocess_image_mlp(image)
                prediction = model_mlp.predict(image_array)
            elif model_choice == 'cnn':
                image_array = preprocess_image_cnn(image)
                prediction = model_cnn.predict(image_array)
            elif model_choice == 'lenet':
                image_array = preprocess_image_cnn(image)
                prediction = model_lenet.predict(image_array)
            else:
                return redirect(request.url)
            
            predicted_digit = np.argmax(prediction, axis=1)[0]
            
            # Render the result
            return render_template('result.html', predicted_digit=predicted_digit, model_choice=model_choice)
    return render_template('upload.html')

# Define a route for the result page
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
