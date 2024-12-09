# Description: This script trains a GAN model for sketch-to-image generation using the provided sketch and photo directories.

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from dataset_loader import create_dataloaders
from models.unet_generator import UNetGenerator
from models.patchgan_discriminator import PatchGANDiscriminator
from evaluate import evaluate_model

# Argument parser for flexible inputs
def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN for sketch-to-image generation.")
    parser.add_argument('--sketch_dir', type=str, default="Data/raw/generate_inverted_sketches", required=False, help='Path to the sketches directory.')
    parser.add_argument('--photo_dir', type=str, default="Data/raw/portraits", required=False, help='Path to the photos directory.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate for generator and discriminator')   
    parser.add_argument('--save_path', type=str, default='savemodels', help='Path to save the trained models.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--experiment_name', type=str, default='gamma_inverted', help='Name of the experiment.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout value for the generator and discriminator.')
    parser.add_argument('--verbose', type=bool, default=True, help='Toggle verbose output.')
    
    return parser.parse_args()


def create_experiment_folder(experiment_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    experiment_folder = os.path.join(save_path, experiment_name) 
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)        
    return experiment_folder

# Main training function
def train(sketch_dir, photo_dir, batch_size, num_epochs, lr, save_path, patience, experiment_name, dropout,verbose=True):
    print("Starting training...")

  


    # Set up device: use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # M1/M2 Macs with Metal support
        print(f"Using device: {device} (MPS)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")

    # Create dataloaders


    train_loader, val_loader, test_loader = create_dataloaders(sketch_dir, photo_dir, batch_size)

    # Initialize models
    generator = UNetGenerator(dropout_value=dropout).to(device)
    discriminator = PatchGANDiscriminator(dropout_value=dropout).to(device)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()  # For adversarial loss
    criterion_L1 = nn.L1Loss()  # For pixel-level similarity

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training settings
    best_val_loss = float("inf")
    patience_counter = 0

    save_path = create_experiment_folder(experiment_name, save_path)
    print(f"Saving models to: {save_path}")

    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        for i, (sketches, real_images) in enumerate(train_loader):

            # Move data to device
            sketches = sketches.to(device)
            real_images = real_images.to(device)

            # === Train Discriminator ===
            # Reset gradients
            optimizer_D.zero_grad()


            # The discriminator tries to classify real images (real_images) and sketches as real (1)
            real_output = discriminator(real_images, sketches) # The pair of sketches and real images are fed to the discriminator
            real_labels = torch.ones_like(real_output).to(device) # Real labels are 1
            loss_real = criterion_GAN(real_output, real_labels) # Loss is calculate based the ability of discriminator to classify the real image as real

            # The generator produces fake images (fake_images) from these sketches. These fake images are paired with the original sketches and fed into the discriminator.
            fake_images = generator(sketches) # generating fake images for the sketches
            fake_output = discriminator(fake_images.detach(), sketches)  # The pair of sketches and the generated image are fed to the discriminator for the discriminator to classify
            fake_labels = torch.zeros_like(fake_output).to(device) # Fake labels are 0
            loss_fake = criterion_GAN(fake_output, fake_labels) # Loss is calculate based the ability of discriminator to classify the fake image as fake

            # Total loss for discriminator
            loss_D = (loss_real + loss_fake) * 0.5 # total loss is the average of the loss of the real and fake images
            loss_D.backward() # Backpropagation
            optimizer_D.step() # Update the weights

    
            optimizer_G.zero_grad() # Reset gradients

            # === Train Generator ===

            # Adversarial loss (tries to fool the discriminator)
            fake_output = discriminator(fake_images, sketches) # The pair of sketches and the generated image are fed to the discriminator for the discriminator to classify.

            loss_G_GAN = criterion_GAN(fake_output, real_labels) # We want to know the descriminators ability to classify the fake-pair as real hence we use the real labels ie :
            # loss = true_class * log(predicted_class) + (1 - true_class) * log(1 - predicted_class), where true_class is 1 and predicted_class is the output of the discriminator
            # The generator goal is make the generator image as real as possible such that discriminator classify the generated images as real (1), and if the discriminator is fooled, the loss will be low else high. This way,
            # generator parameters are updated to generate images that are more realistic by minimizing the loss. Hence we check how well the image is classified as real by the discriminator.

            # Reason for using  L1 loss : 

            """
            Adversarial Loss: Motivates the generator to create images that deceive the discriminator into classifying them as real, fostering overall realism in the generated images.
		    Pixel-wise Loss: Ensures that the generator not only produces believable images but also accurately replicates specific details (like texture, color, and structure) of the target real images, ensuring high fidelity and precision.
            The combination of these losses ensures that the generated images are not only realistic but also closely resemble the target images, capturing the essence and details of the original images.
            """

            loss_G_L1 = criterion_L1(fake_images, real_images) * 100  # The reason for multiplying by 100 is to scale the loss to a reasonable value. This is a hyperparameter that can be tuned.

            # Total loss for generator
            loss_G = loss_G_GAN + loss_G_L1 # The total loss is the sum of the adversarial loss and the L1 loss
            loss_G.backward() # Backpropagation
            optimizer_G.step() # Update the weights

            total_g_loss += loss_G.item() # Accumulate the generator loss [Ability of the generator to generate image that are classified as real pair by the discriminator and the pixel wise deviation from the real image]
            total_d_loss += loss_D.item() # Accumulate the discriminator loss [Ability of the discriminator to classify the real image as real pair and generate image as fake pair]

            if verbose:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # === Validation Phase ===
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_sketches, val_real_images in val_loader:
                val_sketches = val_sketches.to(device)
                val_real_images = val_real_images.to(device)

                # Generate fake images
                val_fake_images = generator(val_sketches)  # Generate fake images from the sketches

                # Calculate validation L1 loss
                val_loss += criterion_L1(val_fake_images, val_real_images).item()  # Calculate the L1 loss between the generated image and the real image; For validation, focusing on pixel-wise L1 loss is typically sufficient and recommended for assessing the quality of the generated images. However, incorporating adversarial loss in some validation checks can provide additional insights into the overall effectiveness of the GAN training process, particularly in how well the generator is improving in terms of generating realistic images

        # if verbose:
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(generator.state_dict(), os.path.join(save_path, "best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(save_path, "best_discriminator.pth"))
            print(f"Model saved at epoch {epoch} with validation loss: {val_loss:.4f} at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Evaluate the best model on the test set
    generator.load_state_dict(torch.load(os.path.join(save_path, "best_generator.pth")))
    generator.eval()

    """
    We aim to to assess how closely the generated images match the target real images in terms of quality and details.
    To evaluate the model, we use two key metrics: Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
    SSIM: SIM is a metric used to measure the similarity between two images. 
    It evaluates how well the structural information is preserved in the generated image compared to the original image, 
    considering aspects like texture, contrast, and luminance. A higher SSIM value (closer to 1) indicates better similarity.
    Here, we aim to genearate visually plausible images hence SSIM provides a more intuitive measure of visual similarity than simpler error metrics like MSE.

    PSNR: PSNR is a metric used to measure the quality of the generated image by comparing it to the original image. PSNR is traditionally used in image processing applications such as compression and restoration, making it a standard benchmark for image quality.
    Higher PSNR means that the generated images have less noise and are more accurate reproductions of the real images.
    
    """

    SSIM, PSNR = evaluate_model(generator, test_loader, device)

    return SSIM, PSNR

# Handle both direct parameter passing and command-line argument passing
def main():
    args = parse_args()
    train(
        sketch_dir=args.sketch_dir,
        photo_dir=args.photo_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_path=args.save_path,
        patience=args.patience,
        experiment_name=args.experiment_name,
        dropout=args.dropout,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()