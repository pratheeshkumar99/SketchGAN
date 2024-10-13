import itertools
from train import train  # Import the train function directly from train.py

# Define hyperparameter grid
learning_rates = [0.0002, 0.0005, 0.001, 0.005]
batch_sizes = [8, 16, 32, 64]
l1_weights = [10, 50, 100, 200]
dropout_rates = [0.1, 0.3, 0.5]

# Best tracking variables
best_ssim = 0
best_psnr = 0
best_params = None

# Create all combinations of the hyperparameters
for lr, batch_size, l1_weight, dropout_rate in itertools.product(learning_rates, batch_sizes, l1_weights, dropout_rates):
    print(f"Running experiment with lr={lr}, batch_size={batch_size}, l1_weight={l1_weight}, dropout_rate={dropout_rate}")
    # Define experiment name based on the hyperparameters
    experiment_name = f"lr_{lr}_batch_{batch_size}_l1_{l1_weight}_dropout_{dropout_rate}"
    print(f"Running experiment: {experiment_name}")


    sketch_dir = "Data/raw/sketches"
    photo_dir =  "Data/raw/portraits"
    batch_size = batch_size
    num_epochs =  50
    lr = lr
    save_path = "savemodels"
    patience = 10
    experiment_name = experiment_name
    dropout = dropout_rate
    verbose = False

    # Directly call the train function from train.py
    ssim, psnr = train(sketch_dir, photo_dir, batch_size, num_epochs, lr, save_path, patience, experiment_name, dropout,verbose=verbose)

    # Check if this is the best model so far
    if ssim > best_ssim:
        best_ssim = ssim
        best_psnr = psnr
        best_params = (lr, batch_size, l1_weight, dropout_rate)
        print(f"New best model found with SSIM: {best_ssim}, PSNR: {best_psnr}, Params: {best_params}")

print(f"Best SSIM: {best_ssim}, PSNR: {best_psnr} with Params: {best_params}")