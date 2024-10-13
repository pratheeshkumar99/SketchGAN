import torch
import random
import argparse
import matplotlib.pyplot as plt
from models.unet_generator import UNetGenerator  # Ensure proper imports from your structure
from dataset_loader import create_dataloaders  # Your dataloader creation method

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Random image selection from test data for visualization')
    parser.add_argument('--photo_dir', type=str, required=True, help='Path to photo directory')
    parser.add_argument('--sketch_dir', type=str, required=True, help='Path to sketch directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the dataloader')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test data split ratio')
    parser.add_argument('--val_split', type=float, default=0.5, help='Validation split from test set')
    parser.add_argument('--checkpoint_path', type=str, default="savemodels/best_generator.pth", help='path of the trained model')
    
    args = parser.parse_args()
    return args

def load_generator(checkpoint_path, device):
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Function to visualize a random image from the test set
def visualize_random_image(test_loader, generator, device):
    # Randomly select a batch from the test_loader
    batch = random.choice(list(test_loader))  # Select a random batch
    
    # Get the sketches and real images from the batch
    sketches, real_images = batch  # Assuming your batch contains both sketches and real images
    
    # Randomly select an index from the current batch
    random_idx = random.randint(0, len(sketches) - 1)
    
    # Select the random sketch and corresponding real image
    random_sketch = sketches[random_idx].unsqueeze(0).to(device)  # Add batch dimension for the model
    random_real_image = real_images[random_idx]  # Get the real image for comparison
    
    # Generate the corresponding photo-like image
    with torch.no_grad():
        generated_image = generator(random_sketch)

    # Unnormalize and visualize the images
    def unnormalize(tensor):
        tensor = tensor * 0.5 + 0.5  # Unnormalize to [0, 1]
        return tensor

    # Convert images to numpy format
    generated_image = unnormalize(generated_image.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    random_real_image = unnormalize(random_real_image).permute(1, 2, 0).cpu().numpy()
    random_sketch_image = unnormalize(random_sketch.squeeze(0)).permute(1, 2, 0).cpu().numpy()

    # Display the random sketch, generated image, and real image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the random original sketch
    axs[0].imshow(random_sketch_image)
    axs[0].axis('off')
    axs[0].set_title("Original Sketch")

    # Display the generated image
    axs[1].imshow(generated_image)
    axs[1].axis('off')
    axs[1].set_title("Generated Image")

    # Display the original real image
    axs[2].imshow(random_real_image)
    axs[2].axis('off')
    axs[2].set_title("Original Real Image")

    plt.show()

def main():
    # Parse arguments
    args = parse_args()

    # Set up device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test dataloader
    _, _, test_loader = create_dataloaders(args.sketch_dir, args.photo_dir, batch_size=args.batch_size)

    # Load the trained generator
    generator = load_generator(args.checkpoint_path, device)

    # Visualize a random image from the test set
    visualize_random_image(test_loader, generator, device)


if __name__ == "__main__":
    main()