import os
import cv2
import torch
import random
import itertools
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes.
    """
    images = []
    labels = []

    for image, label in batch:
        images.append(torch.tensor(image).permute(2, 0, 1))  # Convert image to tensor (C, H, W)
        labels.append(torch.tensor(label))  # Keep labels as a list of bounding boxes

    return images, labels


def add_border(image, labels, low, high):
    """
    Add border to the image, resize it back to the original size,
    and adjust bounding box labels accordingly.

    Args:
        image (np.ndarray): Input image as a NumPy array (H, W, C).
        labels (list): List of bounding boxes in YOLO format (class_id, x_center, y_center, bbox_width, bbox_height).
        low (int): Minimum size of the border.
        high (int): Maximum size of the border.

    Returns:
        np.ndarray, list: Augmented image and adjusted labels.
    """
    # Randomize border sizes
    top = random.randint(low, high)
    bottom = random.randint(low, high)
    left = random.randint(low, high)
    right = random.randint(low, high)

    original_width, original_height = image.shape[1], image.shape[0]

    # Add border to the image
    image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # Resize image back to original size
    resized_image = cv2.resize(image_with_border, (original_width, original_height))

    # Adjust bounding boxes
    new_labels = []
    border_width = left + right
    border_height = top + bottom
    scale_x = original_width / (original_width + border_width)
    scale_y = original_height / (original_height + border_height)

    for class_id, x_center, y_center, bbox_width, bbox_height in labels:
        # Adjust bounding box positions and sizes
        x_center_new = (x_center * original_width + left) / (original_width + border_width)
        y_center_new = (y_center * original_height + top) / (original_height + border_height)
        bbox_width_new = bbox_width * scale_x
        bbox_height_new = bbox_height * scale_y

        new_labels.append((class_id, x_center_new, y_center_new, bbox_width_new, bbox_height_new))

    return resized_image, new_labels


def random_crop(image, labels):
    """
    Crop the image randomly without cutting into any bounding boxes.
    Adjust the bounding boxes accordingly.

    Args:
        image (np.ndarray): Input image as a NumPy array (H, W, C).
        labels (list): List of bounding boxes in YOLO format (class_id, x_center, y_center, bbox_width, bbox_height).

    Returns:
        np.ndarray, list: Cropped and resized image, adjusted bounding boxes.
    """
    original_width, original_height = image.shape[1], image.shape[0]

    # Convert bounding boxes to absolute pixel coordinates
    bboxes_absolute = []
    for class_id, x_center, y_center, bbox_width, bbox_height in labels:
        x_min = (x_center - bbox_width / 2) * original_width
        y_min = (y_center - bbox_height / 2) * original_height
        x_max = (x_center + bbox_width / 2) * original_width
        y_max = (y_center + bbox_height / 2) * original_height
        bboxes_absolute.append((class_id, x_min, y_min, x_max, y_max))

    # Find a valid crop region
    for _ in range(10):  # Try up to 100 attempts to find a valid crop
        x_left = random.randint(0, original_width // 4)
        x_right = random.randint(3 * original_width // 4, original_width)
        y_top = random.randint(0, original_height // 4)
        y_bottom = random.randint(3 * original_height // 4, original_height)

        # Check if the crop region overlaps with any bounding boxes
        is_valid_crop = all(
            x_left <= x_min and x_right >= x_max and y_top <= y_min and y_bottom >= y_max
            for _, x_min, y_min, x_max, y_max in bboxes_absolute
        )

        if is_valid_crop:
            break
    else:
        return image, labels  # Return original image if no valid crop was found

    # Crop the image
    cropped_image = image[y_top:y_bottom, x_left:x_right]

    # Adjust bounding boxes to the new crop region
    adjusted_labels = []
    for class_id, x_min, y_min, x_max, y_max in bboxes_absolute:
        # Adjust coordinates to the cropped image
        x_min_new = x_min - x_left
        y_min_new = y_min - y_top
        x_max_new = x_max - x_left
        y_max_new = y_max - y_top

        # Convert back to YOLO format
        bbox_width = (x_max_new - x_min_new) / (x_right - x_left)
        bbox_height = (y_max_new - y_min_new) / (y_bottom - y_top)
        x_center = (x_min_new + x_max_new) / 2 / (x_right - x_left)
        y_center = (y_min_new + y_max_new) / 2 / (y_bottom - y_top)

        adjusted_labels.append([class_id, x_center, y_center, bbox_width, bbox_height])

    # Resize the cropped image back to the original size
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    return cropped_image, adjusted_labels


def change_brightness(image, label, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img, label


def rotate_image(image, label, range_angle):
    original_width, original_height = image.shape[1], image.shape[0]
    angle = random.randint(-range_angle, range_angle)
    rot_mat = cv2.getRotationMatrix2D((original_width / 2, original_height / 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, (original_width, original_height))

    adjusted_labels = []
    for bbox in label:
        class_id, x_center, y_center, width, height = bbox
        x_center_pixel = x_center * original_width
        y_center_pixel = y_center * original_height
        width_pixel = width * original_width
        height_pixel = height * original_height

        points = [
            [x_center_pixel - width_pixel / 2, y_center_pixel - height_pixel / 2],
            [x_center_pixel + width_pixel / 2, y_center_pixel - height_pixel / 2],
            [x_center_pixel + width_pixel / 2, y_center_pixel + height_pixel / 2],
            [x_center_pixel - width_pixel / 2, y_center_pixel + height_pixel / 2]
        ]
        rotated_points = np.array([cv2.transform(np.array([[pt]], dtype=np.float32), rot_mat)[0][0] for pt in points])

        x_coords = rotated_points[:, 0]
        y_coords = rotated_points[:, 1]

        x_center_new = (max(x_coords) + min(x_coords)) / 2 / original_width
        y_center_new = (max(y_coords) + min(y_coords)) / 2 / original_height
        width_new = (max(x_coords) - min(x_coords)) / original_width
        height_new = (max(y_coords) - min(y_coords)) / original_height

        if width_new > 0 and height_new > 0:
            adjusted_labels.append([class_id, x_center_new, y_center_new, width_new, height_new])

    return rotated_image, adjusted_labels


def save_image_and_labels(image, labels, output_image_path, output_labels_path):
    cv2.imwrite(output_image_path, image)
    with open(output_labels_path, 'w') as label_file:
        for bbox in labels:
            class_id, x_center, y_center, width, height = bbox
            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def resize_image_and_labels(image, labels, target_size):
    """
    Resize image and adjust bounding boxes for the new size.

    Args:
        image (ndarray): Input image.
        labels (list): List of bounding boxes in YOLO format [class_id, x_center, y_center, width, height].
        target_size (tuple): Target size (width, height).

    Returns:
        resized_image (ndarray): Resized image.
        resized_labels (list): Adjusted bounding boxes in YOLO format.
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Resize image
    resized_image = cv2.resize(image, (target_width, target_height))

    # Adjust labels
    resized_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        # Convert x_center, y_center from relative to absolute coordinates
        abs_x_center = x_center * original_width
        abs_y_center = y_center * original_height
        abs_width = width * original_width
        abs_height = height * original_height

        # Resize bounding box
        abs_x_center *= scale_x
        abs_y_center *= scale_y
        abs_width *= scale_x
        abs_height *= scale_y

        # Convert back to relative coordinates
        new_x_center = abs_x_center / target_width
        new_y_center = abs_y_center / target_height
        new_width = abs_width / target_width
        new_height = abs_height / target_height

        resized_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])

    return resized_image, resized_labels


def process_image(image_file, image_dir, label_dir, output_image_dir, output_labels_dir, target_size, augmentations):
    """
    Process a single image file, apply augmentations, and save results.
    """
    try:
        # Read image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        # Read labels
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            return

        with open(label_path, 'r') as file:
            labels = [list(map(float, line.strip().split())) for line in file]

        base_name = os.path.splitext(image_file)[0]

        # Resize image and labels
        resized_image, resized_labels = resize_image_and_labels(image, labels, target_size)
        
        # Save resized image and labels
        output_image_path = os.path.join(output_image_dir, image_file)
        output_labels_path = os.path.join(output_labels_dir, label_file)
        save_image_and_labels(resized_image, resized_labels, output_image_path, output_labels_path)

        # Apply augmentations
        for aug_name, aug_fn in augmentations.items():
            augmented_img, augmented_label = aug_fn(resized_image, resized_labels)
            save_image_and_labels(
                augmented_img, augmented_label,
                os.path.join(output_image_dir, f"{base_name}_{aug_name}.jpg"),
                os.path.join(output_labels_dir, f"{base_name}_{aug_name}.txt"),
            )

        # Apply combinations of augmentations
        for combo_size in range(2, len(augmentations) + 1):
            for combo in itertools.combinations(augmentations.keys(), combo_size):
                augmented_img, augmented_label = resized_image.copy(), resized_labels.copy()
                for aug_name in combo:
                    augmented_img, augmented_label = augmentations[aug_name](augmented_img, augmented_label)
                combo_name = "_".join(combo)
                save_image_and_labels(
                    augmented_img, augmented_label,
                    os.path.join(output_image_dir, f"{base_name}_{combo_name}.jpg"),
                    os.path.join(output_labels_dir, f"{base_name}_{combo_name}.txt"),
                )
    except Exception as e:
        print(f"Error processing {image_file}: {e}")


def add_border_global(img, lbl):
    return add_border(img, lbl, low=10, high=300)

def rotate_image_global(img, lbl):
    return rotate_image(img, lbl, 20)

def random_crop_global(img, lbl):
    return random_crop(img, lbl)

def change_brightness_global(img, lbl):
    return change_brightness(img, lbl, random.randint(-50, 50))


# Augmentations
augmentations = {
    "border": add_border_global,
    "rotate": rotate_image_global,
    "crop": random_crop_global,
    "brightness": change_brightness_global
}


def apply_all_augmentations(image_dir, label_dir, output_image_dir, output_labels_dir, target_size=(416, 416)):
    """
    Apply all augmentations using multiprocessing for faster execution.
    """
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    # List all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.PNG'))]
    
    # Use multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, img_file, image_dir, label_dir, output_image_dir, output_labels_dir, target_size, augmentations)
            for img_file in image_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing dataset"):
            try:
                future.result()  # Raise exceptions if any
            except Exception as e:
                print(f"Error in processing: {e}")
