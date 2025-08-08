import os
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pymysql

app=Flask(__name__)

# Database connection 
db = pymysql.connect( 
host='localhost', 
user='root',
password='root',
database='skin_cancer_db' 
) 

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient_name = request.form['name'] 
        age = int(request.form['age']) 
        weight = float(request.form['weight']) 
        date = request.form['date'] 
        image_file = request.files['image'] 
        
        # Save the image to a specific directory 
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename) 
        image_file.save(image_path)
        
        # Preprocess the image for prediction 
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) 
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension 
        img_array /= 255.0 # Normalize if your model expects normalized inputs 
        
        # Perform the prediction 
        prediction = model.predict(img_array) 
        predicted_class = "benign" if prediction[0][0] < 0.5 else "malignant" 
        probability = prediction[0][0] if predicted_class == "malignant" else 1 - prediction[0][0] 
        
        # Store the prediction in the database 
        cursor = db.cursor() 
        sql = "INSERT INTO predictions (patient_name, age, weight, date, image_path, prediction, probability) VALUES (%s, %s, %s, %s, %s, %s, %s)" 
        cursor.execute(sql, (patient_name, age, weight, date, app.config['UPLOAD_FOLDER'] +'/'+ image_file.filename, 
        predicted_class, probability)) 
        db.commit() 
        return render_template('result.html', 
        patient_name=patient_name, 
        age=age, 
        weight=weight, 
        date=date, 
        prediction=predicted_class, 
        probability=probability, 
        image_path=app.config['UPLOAD_FOLDER'] +'/'+ image_file.filename) 
    except KeyError as e: 
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

if __name__=='__main__':
    app.run(debug=True)