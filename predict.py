import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from controlnet_aux import OpenposeDetector
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenposeDetector model
model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")  # Ensure you set this in your .env file
client = OpenAI()

def extract_face_keypoints(image):
    """Extracts face keypoints from the image using OpenposeDetector."""
    img = cv2.imread(image)
    face_poses = model.detect_poses(img, include_face=True, include_hand=False)
    
    keypoints_list = []
    for pose in face_poses:
        if pose.face is not None:
            keypoints = [(keypoint.x, keypoint.y) for keypoint in pose.face]
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append(None)
    
    return keypoints_list

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the image to a temporary path
    image_path = os.path.join("/tmp", file.filename)
    file.save(image_path)

    # Extract face keypoints
    keypoints = extract_face_keypoints(image_path)
    os.remove(image_path)

    # Load MBTI dataset
    with open("mbti_and_face.json", "r") as file:
        mbti_data = json.load(file)

    # Convert keypoints to JSON
    keypoints_json = json.dumps(keypoints)

    # Call GPT-3 to predict MBTI type
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""You will be provided with a string in json format, representing the face keypoints of a person and you will try to predict the MBTI of the person, \
                              your output will just be the MBTI type of the person and nothing else. \
                              You will make the prediction based on the dataset given to you in the following json string, where the key is the mbti type and the value is a list of face keypoints of that type. \
                              The json string is as follows: {mbti_data}."""
            },
            {
                "role": "user",
                "content": f"You have face keypoints in the following format: {keypoints_json}, please try to predict the MBTI type of this person and only provide the MBTI type (four letters) as the output."
            }
        ],
    )

    mbti_prediction = response.choices[0].message['content']
    return jsonify({"mbti": mbti_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
