import os
import random
import shutil
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths
input_dir = "input_images"  # Directory containing original images
labels_file = "labels.txt"  # Path to your labels.txt file
output_dir = "processed_data"  # Root directory for processed data
training_dir = os.path.join(output_dir, "training")
testing_dir = os.path.join(output_dir, "testing")

# Create output directories if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# INCREASE RESOLUTION
# Define the target resolution (adjust as needed)
target_height = 128  # Recommended for HTR
target_width = 800   # Adjust based on your specific needs

# LABELS
# Function to parse labels file based on the actual format in your labels.txt
def parse_labels(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in labels file: {len(lines)}")
    valid_lines = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        parts = line.split()
        if len(parts) < 8:  # Check for minimum required fields
            print(f"Skipping insufficient data line: {line}")
            continue
            
        word_id = parts[0]
        segmentation_result = parts[1]
        try:
            # Based on your actual file format
            gray_level = int(parts[2])
            
            # Bounding box - directly from your file format
            x, y, w, h = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            
            # The parsing was incorrect before - there is no "number of components" field
            # in your actual data
        except (ValueError, IndexError) as e:
            print(f"Error parsing numeric values in line: {line}")
            print(f"Error details: {e}")
            continue
        
        grammatical_tag = parts[7]
        transcription = " ".join(parts[8:])  # Ensure full transcription is captured

        
        labels[word_id] = {
            'segmentation_result': segmentation_result,
            'gray_level': gray_level,
            'bbox': (x, y, w, h),
            'grammatical_tag': grammatical_tag,
            'transcription': transcription
        }
        valid_lines += 1
    
    print(f"Successfully parsed {valid_lines} label entries")
    if valid_lines > 0:
        print(f"First few label keys: {list(labels.keys())[:5]}")
    return labels

print("Parsing labels file...")
# Parse labels
labels_dict = parse_labels(labels_file)

# Get list of image files
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
print(f"Found {len(image_files)} image files.")
if len(image_files) > 0:
    print(f"Sample image filenames: {image_files[:3]}")

# Try different matching strategies
matched_data = []
print("\nAttempting to match images with labels...")

# Strategy 1: Direct filename matching
print("Strategy 1: Direct filename matching")
for img_file in image_files:
    # Extract the word ID from the image filename (without extension)
    word_id = os.path.splitext(img_file)[0]
    
    if word_id in labels_dict:
        matched_data.append((img_file, labels_dict[word_id]))
        print(f"Matched image {img_file} with label {word_id}")

# If no matches, try alternative strategies
if not matched_data:
    print("\nStrategy 2: Case-insensitive matching")
    label_keys_lower = {k.lower(): k for k in labels_dict.keys()}
    for img_file in image_files:
        word_id_lower = os.path.splitext(img_file)[0].lower()
        if word_id_lower in label_keys_lower:
            original_key = label_keys_lower[word_id_lower]
            matched_data.append((img_file, labels_dict[original_key]))
            print(f"Matched image {img_file} with label {original_key}")

# If still no matches, try more flexible matching
if not matched_data:
    print("\nStrategy 3: Partial matching (image filename contains label ID or vice versa)")
    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        for label_key in labels_dict.keys():
            if img_name in label_key or label_key in img_name:
                matched_data.append((img_file, labels_dict[label_key]))
                print(f"Partial match: image {img_file} with label {label_key}")
                break  # Match each image only once

print(f"\nMatched {len(matched_data)} images with labels.")

# Display your image filenames and label keys to help diagnose the issue
print("\nDiagnostics:")
print(f"Image filenames: {image_files}")
print(f"Label keys: {list(labels_dict.keys())[:10]}")  # Show first 10 keys

# If still no matches, we need to handle this case
if len(matched_data) == 0:
    print("\nNo images could be matched with labels. Taking alternative approach:")
    print("Processing all images without label information")
    
    # Process all images without splitting
    for img_file in image_files:
        # Read the image
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # PREPROCESSING METHODS
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize (maintaining aspect ratio)
        h, w = gray_img.shape
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        
        # Ensure the width is within reasonable bounds
        if new_width > target_width:
            new_width = target_width
        
        resized_img = cv2.resize(gray_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Split into training (80%) and testing (20%) randomly
        if random.random() < 0.8:
            output_dir_for_image = training_dir
        else:
            output_dir_for_image = testing_dir
            
        # Save processed image
        output_path = os.path.join(output_dir_for_image, img_file)
        cv2.imwrite(output_path, resized_img)
    
    print(f"Processed all {len(image_files)} images without label matching.")
    print(f"Images randomly split between training and testing directories.")
    print(f"Training data saved to: {training_dir}")
    print(f"Testing data saved to: {testing_dir}")
    exit(0)  # Exit the script

# If we have matches, continue with the original approach
# Split data into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(matched_data, test_size=0.2, random_state=42)

# Function to process and save data
def process_and_save_data(data, output_directory, dataset_name):
    print(f"\nProcessing {dataset_name} dataset ({len(data)} images)...")
    
    # Create progress bar
    progress_bar = tqdm(total=len(data), unit="image", desc=f"{dataset_name}")
    
    for img_file, label_info in data:
        # Update progress bar
        progress_bar.update(1)
        
        # Read the image
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            progress_bar.write(f"Could not read image: {img_path}")
            continue
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize (maintaining aspect ratio)
        h, w = gray_img.shape
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        
        # Ensure the width is within reasonable bounds
        if new_width > target_width:
            new_width = target_width
        
        resized_img = cv2.resize(gray_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Save processed image
        output_path = os.path.join(output_directory, img_file)
        cv2.imwrite(output_path, resized_img)
        
        # Save corresponding label information
        word_id = os.path.splitext(img_file)[0]
        with open(os.path.join(output_directory, f"{word_id}.txt"), 'w') as f:
            f.write(f"Word ID: {word_id}\n")
            f.write(f"Segmentation Result: {label_info['segmentation_result']}\n")
            f.write(f"Gray Level: {label_info['gray_level']}\n")
            f.write(f"Bounding Box (x,y,w,h): {label_info['bbox']}\n")
            f.write(f"Grammatical Tag: {label_info['grammatical_tag']}\n")
            f.write(f"Transcription: {label_info['transcription']}\n")
    
    # Close progress bar
    progress_bar.close()

# Record start time
start_time = time.time()

# Process and save data
process_and_save_data(train_data, training_dir, "Training")
process_and_save_data(test_data, testing_dir, "Testing")

# Calculate and display total processing time
total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")
print(f"Processed {len(train_data)} training images and {len(test_data)} testing images.")
print(f"Training data saved to: {training_dir}")
print(f"Testing data saved to: {testing_dir}")