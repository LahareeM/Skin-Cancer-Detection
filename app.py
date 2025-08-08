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

# For predict showing result page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient_name = request.form['name'] 
        age = int(request.form['age']) 
        weight = float(request.form['weight']) 
        date = request.form['date'] 
        image_file = request.files['image'] 
        
        # Save the image to a specific directory 
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
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
        image_path=image_path
        ) 
    except KeyError as e: 
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

# For histories showing history page
@app.route('/history') 
def history(): 
    cursor = db.cursor() 
    cursor.execute("SELECT * FROM predictions") 
    predictions = cursor.fetchall() 
    return render_template('history.html', predictions=predictions)

# Reset History will who msg in history page too, after deleting
@app.route('/reset_history', methods=['POST'])
def reset_history():
    cursor = db.cursor() 
    try: 
        # Delete all records from the predictions table 
        cursor.execute("DELETE FROM predictions") 
        db.commit() 
        return render_template('history.html', predictions=[], message="History has been reset.") 
    except Exception as e: 
        db.rollback() # Rollback in case of error 
    return jsonify({"error": str(e)}), 500

# From history page to see perticular report in report page
@app.route('/report/<int:prediction_id>') 
def report(prediction_id): 
    cursor = db.cursor() 
    cursor.execute("SELECT * FROM predictions WHERE id = %s", (prediction_id,)) 
    prediction = cursor.fetchone() 
    if prediction: 
        return render_template('report.html', 
            patient_name=prediction[1], 
            age=prediction[2], 
            weight=prediction[3], 
            date=prediction[4], 
            image_path='/'+prediction[5] , 
            prediction=prediction[6], 
            probability=prediction[7]
            ) 
    else: 
        return "Prediction not found", 404


if __name__=='__main__':
    app.run(debug=True)