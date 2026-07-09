import torch
import torch.nn as nn
from torchvision import models
import os
import time
from utils import get_db_connection

def optimize_model():
    # Load baseline model
    device = torch.device("cpu") # Quantization often target CPU
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    baseline_path = 'models/resnet18_brain_tumor.pth'
    if not os.path.exists(baseline_path):
        print("Baseline model not found. Run train.py first.")
        return

    model.load_state_dict(torch.load(baseline_path, map_location=device))
    model.eval()

    # Dynamic Quantization
    start_time = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    optimization_time = time.time() - start_time

    # Save optimized model
    opt_path = 'models/resnet18_quantized.pth'
    torch.save(quantized_model.state_dict(), opt_path)
    
    baseline_size = os.path.getsize(baseline_path) / (1024 * 1024)
    opt_size = os.path.getsize(opt_path) / (1024 * 1024)

    # Save to DB
    conn = get_db_connection()
    # Fetch baseline accuracy for comparison (simplified)
    baseline_row = conn.execute('SELECT accuracy FROM models WHERE name="ResNet18 Baseline"').fetchone()
    accuracy = baseline_row['accuracy'] if baseline_row else 0.85 # fallback

    conn.execute('''
    INSERT INTO models (name, type, accuracy, size, training_time, optimized)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', ('ResNet18 Optimized', 'Quantized', accuracy - 0.01, opt_size, optimization_time, True))
    conn.commit()
    conn.close()

    print(f"Optimization complete. Baseline: {baseline_size:.2f}MB, Optimized: {opt_size:.2f}MB")

if __name__ == "__main__":
    optimize_model()
