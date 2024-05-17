import requests

def call_api(image_path):
    url = "http://localhost:5000/process_image"
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print("MBTI Prediction:", response.json().get("mbti"))
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    image_path = "test.png"
    call_api(image_path)