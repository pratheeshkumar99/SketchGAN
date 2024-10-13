import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Custom Dataset for loading Sketch and Photo pairs
class SketchPhotoDataset(Dataset):
    def __init__(self, sketch_paths, photo_paths, transform=None):
        self.sketch_paths = sketch_paths
        self.photo_paths = photo_paths
        self.transform = transform

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        # Load the sketch and corresponding photo
        sketch_path = self.sketch_paths[idx]
        photo_path = self.photo_paths[idx]
        
        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")
        
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)

        return sketch, photo

# Create Dataloaders for train, validation, and test splits
def create_dataloaders(sketch_dir, photo_dir, batch_size=16, test_size=0.3, val_split=0.5):
    # Get the list of filenames in each directory
    sketch_files = sorted(os.listdir(sketch_dir))
    photo_files = sorted(os.listdir(photo_dir))

    # Ensure that the two lists have the same length
    assert len(sketch_files) == len(photo_files), "Mismatch between sketch and photo file counts!"

    # Full paths to each sketch and photo
    sketch_paths = [os.path.join(sketch_dir, f) for f in sketch_files]
    photo_paths = [os.path.join(photo_dir, f) for f in photo_files]

    # Now split into train/test sets
    train_sketches, test_sketches, train_photos, test_photos = train_test_split(
        sketch_paths, photo_paths, test_size=test_size, random_state=42
    )

    # Further split test set into validation and test sets
    val_sketches, test_sketches, val_photos, test_photos = train_test_split(
        test_sketches, test_photos, test_size=val_split, random_state=42
    )

    print(f"Training images: {len(train_sketches)}, Validation images: {len(val_sketches)}, Test images: {len(test_sketches)}")

    # Define the transform: Resize the images, convert to tensors, and normalize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize both sketches and photos to 256x256
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Create datasets for train, val, and test sets
    train_dataset = SketchPhotoDataset(train_sketches, train_photos, transform=transform)
    val_dataset = SketchPhotoDataset(val_sketches, val_photos, transform=transform)
    test_dataset = SketchPhotoDataset(test_sketches, test_photos, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader