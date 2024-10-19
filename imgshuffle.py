import os
import shutil

# Define the source directory containing the images
source_directory = r'C:\Users\freed\Downloads\yolo11_suspicious_activity-main\yolo11_suspicious_activity-main\images1'  # Change to your actual source directory

# Define the target directories
normal_folder = r'C:\Users\freed\Downloads\yolo11_suspicious_activity-main\yolo11_suspicious_activity-main\dataset_path\Normal'  # Path to Normal folder
suspicious_folder = r'C:\Users\freed\Downloads\yolo11_suspicious_activity-main\yolo11_suspicious_activity-main\dataset_path\Suspicious'  # Path to Suspicious folder

# Ensure the target directories exist
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(suspicious_folder, exist_ok=True)

# Loop through all files in the source directory
for file_name in os.listdir(source_directory):
    # Get the full path of the file
    file_path = os.path.join(source_directory, file_name)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        # Extract the number from the file name (assuming the format is person_nn_X.jpg)
        if file_name.startswith('person_nn_'):
            # Split the file name to get the number
            number = int(file_name.split('_')[2].split('.')[0])
            
            # Move files based on the number
            if 0 <= number <= 1542:  # Move person_nn_0.jpg to person_nn_10.jpg to Normal folder
                shutil.move(file_path, os.path.join(normal_folder, file_name))
                print(f'Moved {file_name} to Normal folder.')
            else:  # Move person_nn_11.jpg and onwards to Suspicious folder
                shutil.move(file_path, os.path.join(suspicious_folder, file_name))
                print(f'Moved {file_name} to Suspicious folder.')

print("File moving complete.")
