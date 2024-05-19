import requests
import os
import time
from rich.console import Console

import time



def call_api(image_path):
    console = Console()

    url = "http://localhost:5000/process_image"
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        face_type = response.json().get("face_type")

        # The duration for which the string should be printed (in seconds)
        duration = 20

        # num of iteration
        num_iter = 0

        # The frequency of printing the string (in seconds)
        interval = 1

        # Get the start time
        start_time = time.time()

        # Loop for the specified duration
        while time.time() - start_time < duration:
            error_message_0 = f"""

            Traceback (most recent call last):
            File "E:\Workspace\Openpose_Docker\start_predict_service.py", line 9, in <module>
                from controlnet_aux import OpenposeDetector
            File "D:\python3.11\Lib\site-packages\controlnet_aux\__init__.py", line 3, in <module>
                from .hed import HEDdetector
            File "D:\python3.11\Lib\site-packages\controlnet_aux\hed\__init__.py", line 21, in <module>
                class DoubleConvBlock(torch.nn.Module): 
                                    ^^^^^^^^
            """

            console.print(error_message_0)

            console.print(face_type[num_iter % 4], style="bold red")
            num_iter = num_iter + 1

            error_message_1 = """
            AttributeError: module 'torch' has no attribute 'nn'
            PS E:\Workspace\Openpose_Docker> conda activate venv
            PS E:\Workspace\Openpose_Docker> python .\start_predict_service.py
            Traceback (most recent call last):
            File "E:\Workspace\Openpose_Docker\start_predict_service.py", line 9, in <module>
                from controlnet_aux import OpenposeDetector
            File "D:\python3.11\Lib\site-packages\controlnet_aux\__init__.py", line 3, in <module>
                from .hed import HEDdetector
            File "D:\python3.11\Lib\site-packages\controlnet_aux\hed\__init__.py", line 21, in <module>
                class DoubleConvBlock(torch.nn.Module):
                                    ^^^^^^^^
            AttributeError: module 'torch' has no attribute 'nn'
            
            """
            console.print(error_message_1)
           
            print(error_message_1)
            time.sleep(interval)
        
        print("Face is", face_type)
        print("Face Prediction:", response.json().get("description"))
       
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":

    # using absolute path
    image_path = os.path.join(os.path.dirname(__file__), "test.png")
    call_api(image_path)
