# SketchGAN-Sketch-to-Image-Generation-and-Criminal-Identification
Sketch-to-Image Criminal Identification System using a pix2pix GAN trained on the CUHK dataset to generate realistic images from sketches. A classifier is then trained to identify individuals based on generated images, enhancing forensic sketch recognition with deep learning for accurate identification.


# Model Download Instructions

Due to the large size of the trained model files, they are hosted externally on Google Drive. Below you will find the links to download each of the model files. Each model corresponds to a specific set of hyperparameters tested during our experiments.

## Available Models

Here is a list of the available models along with their corresponding hyperparameters and Google Drive download links:

1. **Model with lr=0.005, batch_size=8, l1_weight=100, dropout_rate=0.1**
   - **Description**: This model achieved the highest performance in our tests, showing excellent balance between similarity and noise reduction.
   - **Google Drive Link**: [Download Model](https://drive.google.com/drive/folders/1INPnHaby9jZZUpLsL0K1skMNQ0YX5jbE?usp=drive_link)
   
2. **Model with lr=0.001, batch_size=8, l1_weight=10, dropout_rate=0.5**
   - **Description**: Strong in both SSIM and PSNR metrics, excellent for maintaining high image quality.
   - **Google Drive Link**: [Download Model](https://drive.google.com/drive/folders/1-0lGv7r1zxtROls2EO8GgpKuT0ckxbaa?usp=drive_link)

3. **Model with lr=0.005, batch_size=8, l1_weight=10, dropout_rate=0.5**
   - **Description**: High SSIM score and decent PSNR, indicating good image similarity with reasonable noise reduction.
   - **Google Drive Link**: [Download Model](https://drive.google.com/drive/folders/1ZpMIGWklQqDT6neDr9jRiwI_8xhCe7Zw?usp=drive_link)

### How to Use the Models

After downloading the models, please follow these steps to use them in your projects:

1. **Download the Model**: Click on the link provided above and download the model file to your local machine.
2. **Place the Model in Your Project Directory**: Move the downloaded `.pth` file into the designated model directory in your project structure.
3. **Update Model Path in Code**: Ensure your code references the correct path where the model file is stored.
4. **Load and Use the Model**: Use the standard PyTorch method to load the model weights and evaluate or further fine-tune the model on your data.

### Troubleshooting

If you encounter any issues while downloading or using the models, please check the following:

- Ensure you have sufficient permissions to access the Google Drive links.
- Verify that the downloaded files are complete and not corrupted.
- Check that your environment meets all the dependencies required to run the model.

For more support, feel free to open an issue in this repository or contact [Your Contact Information].
