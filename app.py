import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name=request.form['name']
    age=request.form['age']
    weight=request.form['weight']
    date=request.form['date']
    
    image_file=request.files['image']
    if image_file:
        filename=secure_filename(image_file.filename)
        image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)
        
        # Preprocessing image
        img=image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # To Make Prediction
        prediction=model.predict(img_array)
        predicted_class = "benign" if prediction[0][0] < 0.5 else "malignant"
        probability = prediction[0][0] if predicted_class == "malignant" else 1 - prediction[0][0]
        
        # Return and show result
        return f"""
            Name: {name}<br>
            Age: {age}<br>
            Weight: {weight}<br>
            Date: {date}<br>
            Uploaded Image: {filename}<br><br>
            <b>Prediction:</b> {predicted_class.upper()}<br>
            <b>Confidence:</b> {round(probability * 100, 2)}%
        """
    else:
        return "No image uploaded", 400

if __name__=='__main__':
    app.run(debug=True)