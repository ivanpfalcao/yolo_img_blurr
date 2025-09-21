import os
import random
import shutil

current_file = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file) 

exported_data_dir = f'{current_file_path}/treino/placas_mercosul_tagged'
output_dir = f'{current_file_path}/train_test_labeled_data'

# Define the paths to your exported data

images_dir = os.path.join(exported_data_dir, 'images')
labels_dir = os.path.join(exported_data_dir, 'labels')

# Define the output directories for your train and test sets
train_images_dir = os.path.join(output_dir, 'train/images')
train_labels_dir = os.path.join(output_dir, 'train/labels')
test_images_dir = os.path.join(output_dir, 'test/images')
test_labels_dir = os.path.join(output_dir, 'test/labels')

# Create the output directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get a list of all image files
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

# Define the split ratio (e.g., 80% for training, 20% for testing)
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

# Split the data
train_images = all_images[:split_index]
test_images = all_images[split_index:]

# Function to copy files
def copy_files(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    for image_file in file_list:
        # Copy image
        shutil.copy(os.path.join(source_img_dir, image_file), os.path.join(dest_img_dir, image_file))

        # Copy corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        shutil.copy(os.path.join(source_lbl_dir, label_file), os.path.join(dest_lbl_dir, label_file))

# Copy the files to their respective directories
shutil.copy(os.path.join(exported_data_dir, 'classes.txt'), os.path.join(output_dir, 'classes.txt'))
shutil.copy(os.path.join(exported_data_dir, 'notes.json'), os.path.join(output_dir, 'notes.json'))

copy_files(train_images, images_dir, labels_dir, train_images_dir, train_labels_dir)
copy_files(test_images, images_dir, labels_dir, test_images_dir, test_labels_dir)

print("Data successfully split into training and testing sets!")