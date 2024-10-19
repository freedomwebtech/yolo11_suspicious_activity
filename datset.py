import os
import pandas as pd


df = pd.read_csv(r'C:\Users\freed\Downloads\yolo11_suspicious_activity-main\yolo11_suspicious_activity-main\nkeypoint.csv')

dataset_path = r'C:\Users\freed\Downloads\yolo11_suspicious_activity-main\yolo11_suspicious_activity-main\dataset_path'
sus_path = os.path.join(dataset_path, 'Suspicious')
normal_path = os.path.join(dataset_path, 'Normal')

def get_label(image_name, sus_path, normal_path):
    if image_name in os.listdir(sus_path):
        return 'Suspicious'
    elif image_name in os.listdir(normal_path):
        return 'Normal'
    else:
        return None 

df['label'] = df['image_name'].apply(lambda x: get_label(x, sus_path,normal_path))
df.to_csv(f'{dataset_path}dataset.csv', index=False)