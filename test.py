import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from controlnet_aux import OpenposeDetector
import requests
from io import BytesIO
from PIL import Image

print(torch.cuda.get_device_properties(0).total_memory)

urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
imgs = [
    load_image("https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url)
    for url in urls
]

# download images
for url in urls:
    response = requests.get("https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url)
    img = Image.open(BytesIO(response.content))
    imgs.append(img)
    img.save(url)  # Save the image locally with the same name as the url


model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

poses = [model(img) for img in imgs]

# Assuming poses are PIL Image objects
poses_np = [np.array(pose) if not isinstance(pose, np.ndarray) else pose for pose in poses]

# Create a grid with 2 rows and as many columns as there are images
fig, axs = plt.subplots(2, len(imgs), figsize=(15, 6))

for i, (img, pose) in enumerate(zip(imgs, poses_np)):
    # Display the image in the first row
    axs[0, i].imshow(img)
    axs[0, i].axis('off')  # Hide axes

    # Display the pose in the second row
    axs[1, i].imshow(pose)
    axs[1, i].axis('off')  # Hide axes

# Save the grid to a file
plt.savefig('poses.png')