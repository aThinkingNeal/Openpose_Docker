import requests
import os
import time
from rich.console import Console

def call_api(image_path):
    console = Console()

    url = "http://localhost:5000/process_image"
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        face_type = response.json().get("face_type")

        # The duration for which the string should be printed (in seconds)
        duration = 20

        # The frequency of printing the string (in seconds)
        interval = 1

        # Get the start time
        start_time = time.time()

        # Loop for the specified duration
        num_iter = 0
        while time.time() - start_time < duration:
            error_message_0 = f"""

            Traceback (most recent call last):
            File "E:\\Workspace\\Openpose_Docker\\start_predict_service.py", line 9, in <module>
                from controlnet_aux import OpenposeDetector
            File "D:\\python3.11\\Lib\\site-packages\\controlnet_aux\\__init__.py", line 3, in <module>
                from .hed import HEDdetector
            File "D:\\python3.11\\Lib\\site-packages\\controlnet_aux\\hed\\__init__.py", line 21, in <module>
                class DoubleConvBlock(torch.nn.Module): 
                                    ^^^^^^^^
            """

            error_message_1 = """
            AttributeError: module 'torch' has no attribute 'nn'
            PS E:\\Workspace\\Openpose_Docker> conda activate venv
            PS E:\\Workspace\\Openpose_Docker> python .\\start_predict_service.py
            Traceback (most recent call last):
            File "E:\\Workspace\\Openpose_Docker\\start_predict_service.py", line 9, in <module>
                from controlnet_aux import OpenposeDetector
            File "D:\\python3.11\\Lib\\site-packages\\controlnet_aux\\__init__.py", line 3, in <module>
                from .hed import HEDdetector
            File "D:\\python3.11\\Lib\\site-packages\\controlnet_aux\\hed\\__init__.py", line 21, in <module>
                class DoubleConvBlock(torch.nn.Module):
                                    ^^^^^^^^
            AttributeError: module 'torch' has no attribute 'nn'
            """

            # Printing error messages line by line for smooth scrolling
            for line in error_message_0.splitlines():
                console.print(line + face_type[num_iter % len(face_type)], style="bold red")
                time.sleep(0.05)  # Adjust delay for smoother scrolling

            # Print the face type character highlighted in red
            # console.print(face_type[num_iter % len(face_type)], style="bold red")

            num_iter += 1

            for line in error_message_1.splitlines():
                console.print(line, style="bold red")
                time.sleep(0.05)  # Adjust delay for smoother scrolling

            # time.sleep(interval)
        
        print("Face is", face_type)

        description = response.json().get("description")
        print("Face Prediction:", description)

        output_str = f"""
        
---------------

{face_type}
{description}

---------------
        
        """
        console.print(output_str, style="bold red")


        # Convert the string to bytes
        byte_data = output_str.encode('utf-8')

        # Specify the path to the "123" folder on the Desktop
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        folder_path = os.path.join(desktop_path, '123')
        file_path = os.path.join(folder_path, '123.bin')

        # Ensure the "123" folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Open the file in binary write mode and write the bytes
        with open(file_path, 'wb') as bin_file:
            bin_file.write(byte_data)
       
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    from pathlib import Path 
    # using absolute path
    image_path = os.path.join(os.path.dirname(__file__), "test.png")
    image_name = "bg.png"
    image_path = f"E:\Workspace\out_put\{image_name}"
    image_path = Path(image_path)
    
    call_api(image_path)
