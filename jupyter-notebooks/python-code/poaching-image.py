import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from PIL import Image


resnet_big_weights = models.ResNet101_Weights.DEFAULT

class GarudaDrishti(nn.Module):
    def __init__(self, num_classes, resnet_weights):
        super().__init__()
        self.model = models.resnet101(weights=resnet_weights)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=1)
        
    def forward(self, X: torch.Tensor):
        return self.model(X)


device = "cuda" if torch.cuda.is_available() else "cpu"


poaching_class_list = ['no', 'yes']


poach_transforms_ = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_poaching_on_image(image_path, model, transform, class_names, device):
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        return {'poaching_detected': False, 'details': 'Image file does not exist.'}
    
   
    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)  
    
  
    model.eval()
    with torch.inference_mode():
        outputs = model(input_tensor)
        y_label = torch.round(torch.sigmoid(outputs)).int()
        pred_class = poaching_class_list[y_label.item()] 
    
    print(f"Predicted Class: {pred_class}")
    
    poaching_detected = pred_class.lower() == 'yes'
    
    return {'poaching_detected': poaching_detected}, pil_image

if __name__ == '__main__':
    image_path = 'non-poach-2.jpg' 
    garuda_drishti_model = GarudaDrishti(num_classes=1, resnet_weights=resnet_big_weights).to(device)
    
   
    garuda_drishti_model.load_state_dict(torch.load('garuda_drishtiV1.pth'))

    result, processed_image = detect_poaching_on_image(image_path, garuda_drishti_model, poach_transforms_, poaching_class_list, device)
    
    print(result)
    plt.imshow(processed_image)
    plt.axis('off')
    plt.title(f"Poaching Detected: {result['poaching_detected']}")
    plt.show()
