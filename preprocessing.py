from src import *

import os
import shutil


# Paths to the data
train_image_dir = './data/raw/train/images'
train_label_dir = './data/raw/train/labels'
val_image_dir = './data/raw/valid/images'
val_label_dir = './data/raw/valid/labels'
test_image_dir = './data/raw/test/images'
test_label_dir = './data/raw/test/labels'

train_processed_images_dir = './data/processed/train/images'
train_processed_labels_dir = './data/processed/train/labels'
val_processed_images_dir = './data/processed/valid/images'
val_processed_labels_dir = './data/processed/valid/labels'
test_processed_images_dir = './data/processed/test/images'
test_processed_labels_dir = './data/processed/test/labels'



# Create processed folders if not exist
if not os.path.exists(train_processed_images_dir):
    os.makedirs(train_processed_images_dir)
if not os.path.exists(train_processed_labels_dir):
    os.makedirs(train_processed_labels_dir)
if not os.path.exists(val_processed_images_dir):
    os.makedirs(val_processed_images_dir)
if not os.path.exists(val_processed_labels_dir):
    os.makedirs(val_processed_labels_dir)
if not os.path.exists(test_processed_images_dir):
    os.makedirs(test_processed_images_dir)
if not os.path.exists(test_processed_labels_dir):
    os.makedirs(test_processed_labels_dir)

# Empty the processed folders
for folder in [
        train_processed_images_dir, 
        train_processed_labels_dir, 
        val_processed_images_dir, 
        val_processed_labels_dir, 
        test_processed_images_dir, 
        test_processed_labels_dir
    ]:
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))


# Apply augmentations to the datasets
apply_all_augmentations(
        train_image_dir, 
        train_label_dir, 
        train_processed_images_dir, 
        train_processed_labels_dir,
        target_size=(800, 600)
    )
apply_all_augmentations(
        val_image_dir, 
        val_label_dir, 
        val_processed_images_dir, 
        val_processed_labels_dir,
        target_size=(800, 600)
    )


# Copy the test images and labels to the processed folder
for file in os.listdir(test_image_dir):
    shutil.copy(os.path.join(test_image_dir, file), test_processed_images_dir)
for file in os.listdir(test_label_dir):
    shutil.copy(os.path.join(test_label_dir, file), test_processed_labels_dir)