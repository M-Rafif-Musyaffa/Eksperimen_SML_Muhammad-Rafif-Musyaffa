import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    print(f"[INFO] Memuat data dari {file_path}...")
    return pd.read_csv(file_path)

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df_filtered

def scale_and_prepare(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_ready = X_scaled_df.copy()
    df_ready[target_column] = y.values
    return df_ready

def run_preprocessing_pipeline(input_path, output_path):
    print("=== MEMULAI PROSES PREPROCESSING ===")
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = remove_outliers(df, 'chol')
    df_ready = scale_and_prepare(df, target_column='target')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ready.to_csv(output_path, index=False)
    print(f"[SUKSES] Dataset siap latih disimpan di: {output_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "heart-disease-dataset_raw/heart.csv"
    PROCESSED_DATA_PATH = "preprocessing/heart-disease-dataset_preprocessing/heart_ready.csv"
    
    run_preprocessing_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)