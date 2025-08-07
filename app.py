import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        return f"""
            Received:<br>
            Name: {name}<br>
            Age: {age}<br>
            Weight: {weight}<br>
            Date: {date}<br>
            Saved Image at: {image_path}
        """
    else:
        return "No image uploaded", 400

if __name__=='__main__':
    app.run(debug=True)