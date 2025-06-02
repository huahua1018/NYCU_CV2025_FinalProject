import os
import json
import numpy as np
import shutil
import cv2
from pycocotools.coco import COCO
import argparse
from tqdm import tqdm

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_coco_annotations(annotation_path):
    print(f"Loading annotations from: {annotation_path}")
    try:
        # First try to load the JSON file to check its structure
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            print(f"JSON file loaded successfully. Keys: {data.keys()}")
            
            # Print detailed structure information
            if 'annotations' in data:
                print(f"Type of annotations: {type(data['annotations'])}")
                if isinstance(data['annotations'], dict):
                    print(f"Annotation keys: {list(data['annotations'].keys())[:5]}")
                    first_key = list(data['annotations'].keys())[0]
                    print(f"First annotation entry: {data['annotations'][first_key]}")
                elif isinstance(data['annotations'], list):
                    print(f"Number of annotations: {len(data['annotations'])}")
                    if len(data['annotations']) > 0:
                        print(f"First annotation: {data['annotations'][0]}")
            
            if 'images' in data:
                print(f"Type of images: {type(data['images'])}")
                if isinstance(data['images'], dict):
                    print(f"Image keys: {list(data['images'].keys())[:5]}")
                    first_key = list(data['images'].keys())[0]
                    print(f"First image entry: {data['images'][first_key]}")
                elif isinstance(data['images'], list):
                    print(f"Number of images: {len(data['images'])}")
                    if len(data['images']) > 0:
                        print(f"First image: {data['images'][0]}")
            
            # Create a new COCO format dictionary
            coco_format = {
                'images': data['images'],
                'annotations': [],
                'categories': data['categories'],
                'info': data['info'],
                'licenses': data['licenses']
            }
            
            # Convert annotations from dict to list
            if isinstance(data['annotations'], dict):
                # Sort annotations by image_id
                annotations_list = list(data['annotations'].values())
                annotations_list.sort(key=lambda x: x['image_id'])
                coco_format['annotations'] = annotations_list
            else:
                coco_format['annotations'] = data['annotations']
            
            # Save the converted format
            temp_path = annotation_path + '.temp'
            with open(temp_path, 'w') as f:
                json.dump(coco_format, f)
            
            # Load the converted format
            coco = COCO(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return coco
            
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        raise

def get_cell_type_from_filename(filename):
    # Extract cell type from filename (e.g., "A172_Phase_A3_2_00d04h00m_3.tif" -> "A172")
    return filename.split('_')[0]

def process_livecell_dataset(coco_annotation_path, base_image_dir, output_dir, cell_type=None, all_masks=None, postfix=''):
    # Create output directories
    train_img_dir = os.path.join(output_dir, postfix)
    create_directory(train_img_dir)

    # Load COCO annotations
    print("Loading annotations...")
    coco = load_coco_annotations(coco_annotation_path)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images")
    
    # Use provided dictionary or create new one
    if all_masks is None:
        all_masks = {}
    
    # Process each image with progress bar
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        
        # Get cell type from filename
        cell_type_from_file = get_cell_type_from_filename(img_filename)
        
        # Skip if cell type doesn't match
        if cell_type is not None and cell_type_from_file.lower() != cell_type.lower():
            continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Skip if no annotations
        if len(anns) == 0:
            continue
            
        # Read original image
        img_path = os.path.join(base_image_dir, cell_type_from_file, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue
            
        # Create mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint16)
        instance_id_counter = 1
        
        # Process each annotation
        for ann in anns:
            # Get segmentation mask
            seg = ann['segmentation']
            if isinstance(seg, list):
                # Convert polygon to mask
                polygon = np.array(seg[0]).reshape(-1, 2)
                # Create a temporary binary mask
                temp_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [polygon.astype(np.int32)], 1)
                # Assign instance ID
                mask[temp_mask == 1] = instance_id_counter
                instance_id_counter += 1
        
        # Only save if mask is not empty
        if np.any(mask):
            # Save image
            shutil.copy2(img_path, os.path.join(train_img_dir, img_filename))
            
            # Save mask
            mask_filename = img_filename.replace('.tif', '_seg.npy')
            mask_path = os.path.join(train_img_dir, mask_filename)
            mask_dict = {'masks': mask}
            np.save(mask_path, mask_dict, allow_pickle=True)

            # Store mask in dictionary
            all_masks[str(img_id)] = mask
    
    return all_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process LiveCell dataset for Cellpose training')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'single_cell'], default='full',
                      help='Choose between full LIVECell dataset or single cell type dataset')
    parser.add_argument('--cell_type', type=str, default=None,
                      help='Cell type to process (e.g., a172, bt474, etc.)')
    parser.add_argument('--output_dir', type=str, default='sartorius-cell-instance-segmentation/livecell_processed',
                      help='Directory to save processed dataset')
    parser.add_argument('--base_dir', type=str, default='sartorius-cell-instance-segmentation/LIVECell_dataset_2021',
                      help='Base directory of the LIVECell dataset')
    args = parser.parse_args()

    # Define paths
    base_dir = args.base_dir
    output_dir = args.output_dir
    
    if args.dataset_type == 'single_cell':
        # Process single cell type dataset using cell-specific annotations
        if args.cell_type is None:
            raise ValueError("cell_type must be specified when using single_cell dataset type")
        cell_type = args.cell_type.lower()
        # Create cell-type specific output directory
        output_dir = os.path.join(output_dir, cell_type)
        train_anno_path = os.path.join(base_dir, 'annotations', 'LIVECell_single_cells', cell_type, f'livecell_{cell_type}_train.json')
        val_anno_path = os.path.join(base_dir, 'annotations', 'LIVECell_single_cells', cell_type, f'livecell_{cell_type}_val.json')
        # No need to filter by cell type as we're using cell-specific annotations
        filter_cell_type = None
    else:
        # Process full dataset (all cell types)
        train_anno_path = os.path.join(base_dir, 'annotations', 'LIVECell', 'livecell_coco_train.json')
        val_anno_path = os.path.join(base_dir, 'annotations', 'LIVECell', 'livecell_coco_val.json')
        # No filtering for full dataset
        filter_cell_type = None
    
    base_image_dir = os.path.join(base_dir, 'images', 'livecell_train_val_images')
    
    print("\nProcessing training set...")
    all_masks = process_livecell_dataset(train_anno_path, base_image_dir, output_dir, filter_cell_type, postfix='train')
    
    print("\nProcessing validation set...")
    all_masks = process_livecell_dataset(val_anno_path, base_image_dir, output_dir, filter_cell_type, all_masks, postfix='val')
    
    # Save all masks to a single npy file
    output_path = os.path.join(output_dir, 'all_instance_masks.npy')
    np.save(output_path, all_masks)
    print(f"Saved {len(all_masks)} masks to {output_path}")