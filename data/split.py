import json
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Paths
coco_json_path = "/home/shekhar/identv/D-FINE/data/coco/coco.json"
image_folder = "/home/shekhar/identv/D-FINE/data/coco/train/images"
train_folder = "/home/shekhar/identv/D-FINE/data/train"
val_folder = "/home/shekhar/identv/D-FINE/data/val"
train_json_path = "/home/shekhar/identv/D-FINE/data/train.json"
val_json_path = "/home/shekhar/identv/D-FINE/data/val.json"

# Create folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Load COCO JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

# Group annotations by image
image_to_annotations = defaultdict(list)
for annotation in annotations:
    image_to_annotations[annotation['image_id']].append(annotation)

# Group images by category (based on annotations)
category_to_images = defaultdict(set)
for annotation in annotations:
    category_to_images[annotation['category_id']].add(annotation['image_id'])

# Stratified split
train_images, val_images = [], []
for category_id, image_ids in category_to_images.items():
    image_ids = list(image_ids)
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_images.extend(train_ids)
    val_images.extend(val_ids)

train_images = set(train_images)
val_images = set(val_images)

# Prepare new JSONs
train_coco = {'images': [], 'annotations': [], 'categories': categories}
val_coco = {'images': [], 'annotations': [], 'categories': categories}

for image in images:
    if image['id'] in train_images:
        train_coco['images'].append(image)
        train_coco['annotations'].extend(image_to_annotations[image['id']])
    elif image['id'] in val_images:
        val_coco['images'].append(image)
        val_coco['annotations'].extend(image_to_annotations[image['id']])

# Save new JSON files
with open(train_json_path, 'w') as f:
    json.dump(train_coco, f, indent=4)

with open(val_json_path, 'w') as f:
    json.dump(val_coco, f, indent=4)

# Copy images to respective folders
for image in train_coco['images']:
    src = os.path.join(image_folder, image['file_name'])
    dst = os.path.join(train_folder, image['file_name'])
    shutil.copy(src, dst)

for image in val_coco['images']:
    src = os.path.join(image_folder, image['file_name'])
    dst = os.path.join(val_folder, image['file_name'])
    shutil.copy(src, dst)

print("Data split complete. JSON and folders created.")
