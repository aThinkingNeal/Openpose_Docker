import cv2
import time
from controlnet_aux import OpenposeDetector
             
# Function to extract and save face keypoints
def extract_and_save_face_keypoints(image_path, output_file):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Initialize the OpenposeDetector model
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir="../models")

    # Detect face poses
    face_poses = model.detect_poses(img, include_face=True, include_hand=False)

    # Open the output file
    with open(output_file, 'w') as file:
        for pose in face_poses:
            if pose.face is not None:
                for keypoint in pose.face:
                    file.write(f"({keypoint.x}, {keypoint.y})\n")
            else:
                file.write("No face detected in this pose.\n")

# Main execution
if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    # Define the image path and output file
    image_path = "test.jpg"
    output_file = "face_keypoints.txt"

    # Extract and save face keypoints
    extract_and_save_face_keypoints(image_path, output_file)
    


    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Detection took {duration} seconds.")