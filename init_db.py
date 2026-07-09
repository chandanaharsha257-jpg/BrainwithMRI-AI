import sqlite3

def init_db():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    
    # Patients table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT
    )
    ''')
    
    # Dataset stats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dataset_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        image_count INTEGER,
        avg_width REAL,
        avg_height REAL
    )
    ''')
    
    # Models table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        size REAL,
        training_time REAL,
        optimized BOOLEAN
    )
    ''')
    
    # Predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        filename TEXT,
        result TEXT,
        confidence REAL,
        prediction_time REAL,
        report TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
