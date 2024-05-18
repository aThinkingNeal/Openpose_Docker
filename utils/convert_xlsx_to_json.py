import pandas as pd
import json
import cv2
from openpyxl import load_workbook
import os

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

def get_face_keypoints(image_path):
    img = cv2.imread(image_path)
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir="../models")
    face_poses = model.detect_poses(img, include_face=True, include_hand=False)
    keypoints_list = []
    for pose in face_poses:
        if pose.face is not None:
            # Convert each coordinate to float to ensure JSON serialization
            keypoints = [(float(keypoint.x), float(keypoint.y)) for keypoint in pose.face]
            keypoints_list.append(keypoints)
    return keypoints_list

def process_xlsx_and_save_json(xlsx_path, json_output_path):
    output_dir = 'extracted_images'
    mbti_labels, image_paths = extract_images_and_labels(xlsx_path, output_dir)
    mbti_data = {}

    for label, image_path in zip(mbti_labels, image_paths):
        if label not in mbti_data:  # Only process if label doesn't already exist in mbti_data
            face_keypoints = get_face_keypoints(image_path)
            mbti_data[label] = face_keypoints  # Set the keypoints for the label

    with open(json_output_path, 'w') as json_file:
        json.dump(mbti_data, json_file, indent=4)

# Main execution
if __name__ == "__main__":
    xlsx_path = 'mbti_and_faces.xlsx'
    json_output_path = 'mbti_and_face.json'
    process_xlsx_and_save_json(xlsx_path, json_output_path)