import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Handle both absolute and relative imports
try:
    from ..training.models.unet import UNet
except ImportError:
    # Fallback for absolute import when script is run directly
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from training.models.unet import UNet

def find_model_file(model_dir, epoch):
    model_files = list(Path(model_dir).glob(f'best_model_epoch_{epoch}_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No model file found for epoch {epoch} in directory {model_dir}")
    return model_files[0]

def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1, bilinear=False)  # Use 3 channels for RGB input like training
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def inference(model, image):
    # Step 2: Preprocess the image to match training transforms
    transform = transforms.Compose([
        # Keep RGB format - do NOT convert to grayscale since models expect 3 channels
        transforms.ToTensor(),         # Convert to a PyTorch tensor (RGB)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension



    # Wykonaj inferencję
    with torch.no_grad():
        output = model(input_tensor)

    # Apply appropriate post-processing based on number of output channels
    num_output_channels = output.shape[1]
    if num_output_channels == 1:
        # Binary segmentation
        output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        print(f"Applied binary segmentation post-processing (sigmoid)")
    else:
        # Multi-class semantic segmentation
        output = torch.softmax(output, dim=1)  # Apply softmax for multi-class
        output = torch.argmax(output, dim=1, keepdim=True).float()  # Get class predictions
        print(f"Applied multi-class segmentation post-processing (softmax + argmax) for {num_output_channels} classes")
    
    output_image = output.squeeze().cpu().numpy()  # Convert back to numpy for visualization
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
    output_image = output_image.astype('uint8')
    print("Output min:", output_image.min())
    print("Output max:", output_image.max())
    print("Output mean:", output_image.mean())

    return output_image

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_dir', type=str, default='data/models', help='Directory containing the trained models')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the model to load')
    args = parser.parse_args()

    # Znajdź odpowiedni plik modelu na podstawie numeru epoki
    model_path = find_model_file(args.model_dir, args.epoch)
    print(f"Loading model from {model_path}")

    # Załaduj model
    model = load_model(model_path)


    # Wczytaj obraz
    image = Image.open(args.image_path).convert('RGB')

    # Wykonaj inferencję
    output_image = inference(model, image)
    
    # image = Image.open(args.image_path).convert('RGB')

    # Wizualizacja wyników (opcjonalnie)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Model Output")
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()