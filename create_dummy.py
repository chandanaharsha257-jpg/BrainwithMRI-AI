import torch
import torch.nn as nn
from torchvision import models
import os
from utils import get_db_connection

def create_dummy_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), 'models/resnet18_brain_tumor.pth')
    
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO models (name, type, accuracy, precision, recall, f1, size, training_time, optimized)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ResNet18 Baseline', 'CNN', 0.92, 0.91, 0.92, 0.91, 44.7, 1200, False))
    conn.commit()
    conn.close()
    print("Dummy model created and registered.")

if __name__ == "__main__":
    create_dummy_model()
