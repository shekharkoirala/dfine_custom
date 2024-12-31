import json
import os
from PIL import Image


def validate_images(coco_json_path, images_dir, output_json_path):
    """
    Validate images in a COCO dataset, remove problematic ones, and count them.
    Args:
        coco_json_path (str): Path to the COCO JSON file.
        images_dir (str): Directory where the images are stored.
        output_json_path (str): Path to save the cleaned COCO JSON file.
    Returns:
        int: Count of problematic images.
    """
    # Load COCO dataset
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    valid_images = []
    problematic_images = []
    problematic_count = 0

    # Check each image
    for image in coco_data['images']:
        image_path = os.path.join(images_dir, image['file_name'])
        try:
            with Image.open(image_path) as img:
                img.verify()  # Validate the image
                valid_images.append(image)
        except (OSError, IOError):
            problematic_images.append(image['id'])
            problematic_count += 1

    # Filter annotations that refer to valid images
    valid_image_ids = {img['id'] for img in valid_images}
    valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]

    # Update the dataset
    coco_data['images'] = valid_images
    coco_data['annotations'] = valid_annotations

    # Save the cleaned dataset
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Processed {len(coco_data['images']) + problematic_count} images. Found {problematic_count} problematic images.")
    return problematic_count

# Paths to your datasets and image directories
train_json_path = 'train.json'
train_images_dir = './train/'  # Replace with your train images directory
output_train_json = 'train.json'

val_json_path = 'val.json'
val_images_dir = './val/'  # Replace with your val images directory
output_val_json = 'val.json'

# Validate train.json
print("Validating train.json...")
problematic_train_count = validate_images(train_json_path, train_images_dir, output_train_json)
print(f"Problematic images in train.json: {problematic_train_count}")

# Validate val.json
print("Validating val.json...")
problematic_val_count = validate_images(val_json_path, val_images_dir, output_val_json)
print(f"Problematic images in val.json: {problematic_val_count}")
