<<<<<<< HEAD
# NeuroScan AI: Medical MRI Brain Tumor Classification

An end-to-end Medical Imaging AI web application for MRI Brain Tumor classification and report generation.

## Features
- **Dataset Analysis**: Automated image counting and distribution plotting.
- **CNN Pipeline**: ResNet18 training with augmentation and metric tracking.
- **Optimization Engine**: INT8 Dynamic Quantization for edge deployment.
- **Groq Integration**: Automated radiology report generation using Llama-3.
- **Modern Dashboard**: Dark-themed UI with Chart.js analytics.

## Tech Stack
- **Backend**: Python Flask, PyTorch, SQLite3
- **Frontend**: HTML5, Jinja2, Bootstrap 5, Chart.js
- **AI**: ResNet18 (CNN), Groq API (LLM)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize Database:
   ```bash
   python init_db.py
   ```

3. Analyze Dataset:
   ```bash
   python analyze_dataset.py
   ```

4. Train Model (Optional - Pre-trained logic included):
   ```bash
   python train.py
   ```

5. Optimize Model:
   ```bash
   python optimize.py
   ```

6. Run Application:
   ```bash
   python app.py
   ```

## Folder Structure
- `app.py`: Main Flask application.
- `train.py`: Training pipeline.
- `analyze_dataset.py`: Dataset analysis module.
- `optimize.py`: Model optimization engine.
- `groq_client.py`: Groq API integration.
- `utils.py`: Shared utilities.
- `models/`: Saved model weights.
- `templates/`: Jinja2 templates.
- `static/`: CSS and generated plots.
- `uploads/`: Temporary storage for uploaded scans.
=======
# BrainwithMRI-AI
A deep learning-based web application for brain tumor detection using Python, Flask, TensorFlow, and OpenCV.
>>>>>>> e4c7e35afac2f14da7f2441e5b64f257eb25d332
