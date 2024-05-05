from openai import OpenAI
import json
from dotenv import load_dotenv
import os

# set up the OpenAI API KEY environment variable
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

client = OpenAI()

# Load the GPT-3 API key from an environment variable or secret management service
api_key = "sk-proj-mxoh8TUnFtYlXlO9psYyT3BlbkFJH1vWArIKfsLjtvHws6H4"  # Replace with your API key

# load the face keypoints of person to detect from the txt file
face_keypoints = []

with open("face_keypoints.txt", "r") as file:
    for line in file:
        if "No face detected" in line:
            face_keypoints.append(None)
        elif line.startswith("("):  # Add this condition to avoid converting strings like "No face detected" to float
            keypoints = [tuple(map(float, point.strip("()").split(", ")))
                         for point in line.strip().strip("()").split(") (")]
            face_keypoints.append(keypoints)

# Convert the face keypoints to a JSON string
face_keypoints_json = json.dumps(face_keypoints)

# print(face_keypoints_json) for debugging 

# load the dataset of face keypoints and the corresponding mbti from the json file
with open("mbti_and_face.json", "r") as file:
    mbti_data = json.load(file)


print(mbti_data)


# Give the JSON string to the GPT model to generate a response
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
      "content": f"You have face keypoints in the following format: {face_keypoints_json}, please try to predict the MBTI type of this person and only provide the MBTI type (four letters) as the output."
    }
  ],
)

print(response.choices[0].message)