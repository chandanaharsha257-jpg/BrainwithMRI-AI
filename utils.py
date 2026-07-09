import sqlite3
import os
import torch
from torchvision import transforms
from PIL import Image

def get_db_connection():
    conn = sqlite3.connect('db.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def get_classes():
    return ['glioma', 'meningioma', 'notumor', 'pituitary']

def save_prediction(patient_id, filename, result, confidence, prediction_time, report):
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO predictions (patient_id, filename, result, confidence, prediction_time, report)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, filename, result, confidence, prediction_time, report))
    conn.commit()
    conn.close()
