import requests
import os


def call_api(image_path):
    url = "http://localhost:5000/process_image"
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print("MBTI is", response.json().get("mbti"))
        print("Portrait is", response.json().get("portrait"))
        print("Face Prediction:", response.json().get("description"))
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":

    # using absolute path
    image_path = os.path.join(os.path.dirname(__file__), "test.png")
    call_api(image_path)
