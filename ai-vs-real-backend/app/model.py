import torch
from .hybrid_model import HybridViTCNNClassifier
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationModel:
    def __init__(self, model_path: str):
        self.model = HybridViTCNNClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor.to(device))
            prob = output.item()
            label = True if prob > 0.5 else False
            return label, round(prob, 3)

class CaptioningModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

    def generate_caption(self, image):
        inputs = self.processor(image, return_tensors="pt").to(device)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)