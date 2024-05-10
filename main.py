import cv2
import time
from controlnet_aux import OpenposeDetector
import os
import numpy as np
import matplotlib.pyplot as plt
             
# Function to extract and save face keypoints
def extract_and_save_face_keypoints(image_path, output_file):

    imgs = []

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    imgs.append(img)

    # Initialize the OpenposeDetector model
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir="../models")

    # Detect face poses
    face_poses = model.detect_poses(img, include_face=True, include_hand=False)

    # delete the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Open the output file
    with open(output_file, 'w') as file:
        for pose in face_poses:
            if pose.face is not None:
                for keypoint in pose.face:
                    file.write(f"({keypoint.x}, {keypoint.y})\n")
            else:
                file.write("No face detected in this pose.\n")
    
    poses = [model(img,include_face = True) for img in imgs]

    poses_np = [np.array(pose) if not isinstance(pose, np.ndarray) else pose for pose in poses]
    
    # Create a grid with 2 rows and as many columns as there are images
    fig, axs = plt.subplots(2, len(face_poses), figsize=(15, 6))

    for i, (img, pose) in enumerate(zip(imgs, poses_np)):
        # Check if axs is 1-dimensional
        if len(face_poses) == 1:
            ax_img = axs[0]
            ax_pose = axs[1]
        else:
            ax_img = axs[0, i]
            ax_pose = axs[1, i]

        # Display the image in the first row
        ax_img.imshow(img)
        ax_img.axis('off')

        # Display the pose in the second row
        ax_pose.imshow(pose)
        ax_pose.axis('off')
        # Save the grid to a file
        plt.savefig('poses.png')

        # save the pose as a single png file
        plt.imsave(f'{mbti_str}_pose.png',pose)
    
    


# Main execution
if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    mbti_str = "infj"

    # Define the image path and output file
    image_path = f"{mbti_str}.png"
    output_file = "face_keypoints.txt"

    # Extract and save face keypoints
    extract_and_save_face_keypoints(image_path, output_file)
    

    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Detection took {duration} seconds.")