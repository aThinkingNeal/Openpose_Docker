import requests
import os
import time
from rich.console import Console
from pathlib import Path
import time 


def call_api(image_path):

    start_time = time.time()
    console = Console()
    
    url = "http://localhost:5000/process_image"
    files = {'image': open(image_path, 'rb')}
    
    response = requests.post(url, files=files)

    end_time = time.time()

    ai_process_time = end_time - start_time

    # print the time taken to process the image
    # print(f"Time taken to process the image: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        face_type = response.json().get("face_type")

        # The duration for which the string should be printed (in seconds), now it is 8 seconds
        # 举例 目前处理时间为12秒，打印信息时间为8秒，总时间为12+8=20秒
        # 如果希望将总时间改为25秒，那么将下方duration改为25 - 12 = 13秒即可
        duration = 6

        # The frequency of printing the string (in seconds)
        interval = 1

        # Get the start time
        start_time = time.time()

        # Loop for the specified duration
        num_iter = 0
        while time.time() - start_time < duration:
            # error_message_0 = f""""""

            # # Printing error messages line by line for smooth scrolling
            # for line in error_message_0.splitlines():
            #     # if num_iter % 3 == 0:
            #     #    console.print(line, style= "green")
            #     # else:
            #     console.print(line, style="bold red")
            #     time.sleep(2)  # Adjust delay for smoother scrolling

            # Print the face type character highlighted in red
            num_iter += 1

        # print("Face is", face_type)

        description = response.json().get("description")
        # print("Face Prediction:", description)

        output_str = description
        
        console.print(output_str, style="bold red")


        # Convert the string to bytes
        byte_data = output_str.encode('utf-8')

        # Specify the path to the "123" folder on the Desktop
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        folder_path = os.path.join(desktop_path, '123')
        file_path = os.path.join(folder_path, '123.bin')

        # Ensure the "123" folder exists
        os.makedirs(folder_path, exist_ok=True)

        # print the ai process time
        #print(f"Time taken to process the image: {ai_process_time:.2f} seconds")

        # Open the file in binary write mode and write the bytes
        with open(file_path, 'wb') as bin_file:
            bin_file.write(byte_data)
       
    else:
        print("Error:", response.status_code, response.text)

def monitor_file(image_path):
    
    last_modification_time = os.path.getmtime(image_path)
    while True:
        start_time = time.time()
        current_modification_time = os.path.getmtime(image_path)
        if current_modification_time != last_modification_time:
            last_modification_time = current_modification_time
            call_api(image_path)
        end_time = time.time()
        # print(f"Time taken to for the whole process: {end_time - start_time:.2f} seconds")
        time.sleep(1)

if __name__ == "__main__":
    
    image_name = "bg1.png"
    image_path = f"E:\\Workspace\\out_put\\{image_name}"
    image_path = Path(image_path)
    
    monitor_file(image_path)
    

    
