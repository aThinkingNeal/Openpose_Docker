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
import random
from rich.console import Console

from pathlib import Path
import sys

# Get the current script's directory
current_dir = Path(__file__).resolve().parent

# Get the parent directory
parent_dir = current_dir.parent

# Insert the parent directory into sys.path
sys.path.insert(0, str(parent_dir))

from utils import PORTRAIT_DICT, MBTI_TO_FACE_DICT,PREDICTION_DICT


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
            # Convert keypoints to native Python floats
            keypoints = [(float(keypoint.x), float(keypoint.y)) for keypoint in pose.face]
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append(None)
    
    return keypoints_list

@app.route('/process_image', methods=['POST'])
def process_image():

    console = Console()

    # add start time
    start_time = time.time()

    # add log info that the process is started
    console.print("The process is started", style="bold green")


    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Define a temporary directory relative to the location of the code file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the image to the temporary path
    image_path = os.path.join(temp_dir, file.filename)
    file.save(image_path)

    # Extract face keypoints
    keypoints = extract_face_keypoints(image_path)
    os.remove(image_path)

    # Load MBTI dataset using absolute path
    json_file_path = os.path.join(base_dir, "mbti_and_face.json")
    with open(json_file_path, "r") as file:
        mbti_data = json.load(file)

    # Convert keypoints to JSON
    keypoints_json = json.dumps(keypoints)

    # Call GPT-4o to predict MBTI type
    response = client.chat.completions.create(
        model="gpt-4o",
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

    mbti_prediction = response.choices[0].message.content

    mbti_upper = str(mbti_prediction).upper()

    print("MBTI is: ", mbti_upper)

    portrait = PORTRAIT_DICT[mbti_upper]
    face_type = MBTI_TO_FACE_DICT[mbti_upper]

    print(f"The portraits are: {portrait}")

    # Call GPT-4o to describe the face
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""现在你面前有一个人，你将会被给予一段描述，你需要将这段描述的第二句话进行润色和增添，同时维持其他描述的文字不变"""
            },
            {
                "role": "user",
                "content": f"面部描述为： {portrait}，你需要将这段面部描述的第二句话进行润色和增添，同时维持其他描述的文字不变." ,
            }
        ],
    )

    description = response.choices[0].message.content

    
    prediction_dict = PREDICTION_DICT[mbti_upper]

    # randomly select a prediction key
    prediction_key = random.choice(list(prediction_dict.keys()))
    prediction = prediction_dict[prediction_key]

    # Call GPT-4o to give the prediction
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""现在你将被给予一段运势预测，你需要基于这段预测内容，向对方描述他的运势，你的输出将是对方的运势描述，你需要做部分语言的替换和增减。"""
            },
            {
                "role": "user",
                "content": f"对方的面相为{face_type}，运势预测为{prediction}，请输出一段完整的运势预测，并且让你的语言尽可能自然，而不只是复制运势预测的原话，基于文本给出你自己的独特版本。" ,
            }
        ],
    )

    
    final_output= f"""


您的面相分析：{face_type}----{description}

-------------------------------
-------------------------------

您的未来运势：{prediction_key}----{prediction}
-------------------------------



    """

    # add end time
    end_time = time.time()

    # log the process time
    console.print(f"Time taken to process the image: {end_time - start_time:.2f} seconds", style="bold green")

    return jsonify({
        "face_type": face_type,
        "description": final_output,
    })

if __name__ == '__main__':


    app.run(host='0.0.0.0', port=5000, debug=True)

