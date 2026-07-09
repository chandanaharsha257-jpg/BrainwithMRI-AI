import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from utils import get_db_connection

def train_model(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
    test_dataset = datasets.ImageFolder('dataset/test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = 'models/resnet18_brain_tumor.pth'
    torch.save(model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024) # MB
    
    # Save metrics to DB
    conn = get_db_connection()
    conn.execute('''
    INSERT INTO models (name, type, accuracy, precision, recall, f1, size, training_time, optimized)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ResNet18 Baseline', 'CNN', accuracy, precision, recall, f1, model_size, training_time, False))
    conn.commit()
    conn.close()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'])
    plt.title('Training Accuracy')
    plt.savefig('static/training_plots.png')
    plt.close()
    
    print("Training complete and metrics saved.")

if __name__ == "__main__":
    train_model(epochs=1) # Just 1 epoch for demo/test purposes
