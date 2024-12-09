'''
# This script generates gamma-inverted sketches from the original photos and trains the GAN model using the generated sketches.
# The script uses OpenCV to convert the images to gamma-inverted sketches and trains the model using the generated sketches.
# The training script is called as an external process using the `subprocess` module.

'''
import os
import cv2
import numpy as np
import subprocess
import argparse

from train import train

# Define paths
photo_dir = "/Users/pratheeshjp/Documents/SketchGAN-Sketch-to-Image-Generation-and-Criminal-Identification/Data/raw/portraits"  # Directory containing original photos
sketch_dir = "/Users/pratheeshjp/Documents/SketchGAN-Sketch-to-Image-Generation-and-Criminal-Identification/Data/raw/gamma_inverted_sketches"  # Directory to store generated sketches



#The image  is converted to a gamma-inverted sketch using the following steps:
# Load the image using OpenCV
# Convert the image to grayscale
# Apply Gaussian blur to the grayscale image
# Divide the grayscale image by the blurred image to enhance the edges ( This enchnaces the edges of the image)
# Apply gamma correction with inversion to the enhanced image : gamma inversion can help the GAN to better learn and reproduce these features in its outputs,; 
#This would  inverting the light-dark dynamics enhances contrast, particularly around edges and texture making the lighe areas darker and dark areas lighter which can help the GAN to better learn and reproduce these features in its outputs
# Save the gamma-inverted sketch
def render_sketch(image_path, output_path):
    """
    Function to convert an image to a gamma-inverted sketch.
    """
    # Load image
    img_rgb = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21,21), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)

    # Apply gamma correction with inversion
    gamma = 0.1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_blend, table)

    # Save the sketch
    status = cv2.imwrite(output_path, img_gamma)
    print(status)

def generate_sketches():
    """
    Process all images in the photo directory to generate sketches.
    """
    for filename in os.listdir(photo_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            photo_path = os.path.join(photo_dir, filename)
            sketch_path = os.path.join(sketch_dir, f'sketch_{filename}')
            render_sketch(photo_path, sketch_path)
            print(f"Generated sketch for {filename}")

# def train_model():
#     """
#     Calls the external Python script to train the model using generated sketches.
#     """
#     subprocess.run(['python', train_script_path], check=True)

def main():
    print("Generating gamma-inverted sketches...")
    generate_sketches()
    print("Starting model training...")
    # train_model()
    # print("Model training completed.")

if __name__ == "__main__":
    main()