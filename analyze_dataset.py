import os
import sqlite3
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_db_connection

def analyze_dataset(data_dir='dataset/train'):
    classes = os.listdir(data_dir)
    stats = []
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
            
        images = os.listdir(cls_path)
        count = len(images)
        widths, heights = [], []
        
        for img_name in images[:50]: # Sample 50 for speed
            try:
                with Image.open(os.path.join(cls_path, img_name)) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                continue
        
        avg_w = sum(widths) / len(widths) if widths else 0
        avg_h = sum(heights) / len(heights) if heights else 0
        
        stats.append({
            'class_name': cls,
            'image_count': count,
            'avg_width': avg_w,
            'avg_height': avg_h
        })
    
    # Save to DB
    conn = get_db_connection()
    conn.execute('DELETE FROM dataset_stats')
    for s in stats:
        conn.execute('''
        INSERT INTO dataset_stats (class_name, image_count, avg_width, avg_height)
        VALUES (?, ?, ?, ?)
        ''', (s['class_name'], s['image_count'], s['avg_width'], s['avg_height']))
    conn.commit()
    conn.close()
    
    # Plotting
    df = pd.DataFrame(stats)
    plt.figure(figsize=(10, 6))
    plt.bar(df['class_name'], df['image_count'], color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('static/class_distribution.png')
    plt.close()
    
    print("Dataset analysis complete.")

if __name__ == "__main__":
    analyze_dataset()
