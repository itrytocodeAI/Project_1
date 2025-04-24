from PIL import Image
import torchvision.transforms as transforms

def preprocess_for_classification(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)
