import pandas as pd
import json
import cv2
from openpyxl import load_workbook
import os
import numpy as np
import matplotlib.pyplot as plt
import time


# Assuming controlnet_aux and OpenposeDetector are correctly set up
from controlnet_aux import OpenposeDetector

def extract_images_and_labels(xlsx_path, output_dir):
    # Load the workbook and the active worksheet
    wb = load_workbook(filename=xlsx_path)
    ws = wb.active

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    for i, drawing in enumerate(ws._images):
        # Assuming images are PNG. Adjust if necessary.
        image_path = os.path.join(output_dir, f'image_{i}.png')
        with open(image_path, 'wb') as img_file:
            img_file.write(drawing._data())
        image_paths.append(image_path)

    # Read MBTI labels using pandas for convenience
    df = pd.read_excel(xlsx_path, usecols=['MBTI_Label'])
    # Convert each MBTI label to lowercase
    mbti_labels = [label.lower() for label in df['MBTI_Label'].tolist()]

    return mbti_labels, image_paths

def extract_and_save_face_keypoints(mbti_labels, image_paths, output_dir):
    for mbti_str, image_path in zip(mbti_labels, image_paths):
        imgs = []
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        imgs.append(img)
        model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir="../models")
        face_poses = model.detect_poses(img, include_face=True, include_hand=False)
        if not face_poses:
            print(f"No face poses detected in image {image_path}. Skipping...")
            continue
        poses = [model(img,include_face = True, include_body=False, include_hand=False) for img in imgs]
        poses_np = [np.array(pose) if not isinstance(pose, np.ndarray) else pose for pose in poses]
        fig, axs = plt.subplots(2, len(face_poses), figsize=(15, 6))
        for i, (img, pose) in enumerate(zip(imgs, poses_np)):
            if len(face_poses) == 1:
                ax_img = axs[0]
                ax_pose = axs[1]
            else:
                ax_img = axs[0, i]
                ax_pose = axs[1, i]
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_pose.imshow(pose)
            ax_pose.axis('off')
        plt.savefig(os.path.join(output_dir, f'{mbti_str}_poses.png'))


if __name__ == "__main__":
    start_time = time.time()
    xlsx_path = 'mbti_and_faces.xlsx'
    output_dir = 'mbti_poses'
    mbti_labels, image_paths = extract_images_and_labels(xlsx_path, output_dir)
    extract_and_save_face_keypoints(mbti_labels, image_paths, output_dir)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Detection took {duration} seconds.")