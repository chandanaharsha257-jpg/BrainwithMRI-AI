from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import time
import torch
import torch.nn as nn
from torchvision import models
from werkzeug.utils import secure_filename
from utils import get_db_connection, preprocess_image, get_classes, save_prediction
from groq_client import generate_medical_report
import sqlite3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load Model
device = torch.device("cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model_path = 'models/resnet18_brain_tumor.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    stats = conn.execute('SELECT * FROM dataset_stats').fetchall()
    models_info = conn.execute('SELECT * FROM models').fetchall()
    recent_predictions = conn.execute('''
        SELECT p.*, pt.name as patient_name 
        FROM predictions p 
        LEFT JOIN patients pt ON p.patient_id = pt.id 
        ORDER BY p.id DESC LIMIT 10
    ''').fetchall()
    conn.close()
    return render_template('dashboard.html', stats=stats, models=models_info, predictions=recent_predictions)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        file = request.files.get('file')
        
        if file and name:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Save Patient
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO patients (name, age, gender) VALUES (?, ?, ?)', (name, age, gender))
            patient_id = cur.lastrowid
            conn.commit()
            conn.close()
            
            return redirect(url_for('predict', patient_id=patient_id, filename=filename))
            
    return render_template('upload.html')

@app.route('/predict/<int:patient_id>/<filename>')
def predict(patient_id, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Run Prediction
    input_tensor = preprocess_image(file_path)
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    prediction_time = time.time() - start_time
    
    classes = get_classes()
    result = classes[predicted_idx.item()]
    conf_val = confidence.item()
    
    # Generate Groq Report
    report = generate_medical_report(result, conf_val)
    
    # Save Prediction
    save_prediction(patient_id, filename, result, conf_val, prediction_time, report)
    
    return render_template('result.html', result=result, confidence=conf_val, time=prediction_time, report=report, filename=filename)

@app.route('/history')
def history():
    conn = get_db_connection()
    history_data = conn.execute('''
        SELECT p.*, pt.name as patient_name, pt.age, pt.gender
        FROM predictions p
        JOIN patients pt ON p.patient_id = pt.id
        ORDER BY p.id DESC
    ''').fetchall()
    conn.close()
    return render_template('history.html', history=history_data)

@app.route('/analyze_dataset_route')
def analyze_dataset_route():
    import subprocess
    subprocess.run(['python', 'analyze_dataset.py'])
    return redirect(url_for('dashboard'))

@app.route('/train_route')
def train_route():
    import subprocess
    subprocess.run(['python', 'train.py'])
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
