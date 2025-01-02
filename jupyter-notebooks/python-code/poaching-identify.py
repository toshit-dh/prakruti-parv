import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from PIL import Image
import cv2
import random



resnet_big_weights=models.ResNet101_Weights.DEFAULT
class GarudaDrishti(nn.Module):
    def __init__(self,num_classes,resnet_weights):
        super().__init__()
        self.model=models.resnet101(weights=resnet_weights)
        self.model.fc=nn.Linear(in_features=self.model.fc.in_features,out_features=1)
        
    def forward(self,X:torch.Tensor):
        return self.model(X)

device="cuda" if torch.cuda.is_available() else "cpu"

poaching_class_list=['no','yes']
poach_transforms_=transforms.Compose([
    transforms.Resize((224, 224)),  
    #transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def detect_poaching_on_video(video_path, model, transform, class_names, device, max_frames=300):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return {'poaching_detected': False, 'details': 'Video file does not exist.'}
    
    cap = cv2.VideoCapture(video_path)
    detected_frames = []
    yes_list = []
    no_list = []
    
    if not cap.isOpened():
        print(f"Failed to open video file {video_path}.")
        return {'poaching_detected': False, 'details': 'Failed to open video file.'}
    
    frame_count = 0
    print(f"Starting poaching detection on video: {video_path}")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or failed to read the frame.")
            break
        
        frame_count += 1
        if frame_count % 10 ==0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame).convert("RGB")
            
            input_tensor = poach_transforms_(pil_image).unsqueeze(0).to(device) # type: ignore
            
            model.eval()
            with torch.inference_mode():
                outputs = model(input_tensor)
                y_label=torch.round(torch.sigmoid(outputs)).int()
                pred_class = poaching_class_list[y_label] # type: ignore
            
            if pred_class.lower() == 'yes':
                yes_list.append('yes')
                detected_frames.append(pil_image)
            else:
                no_list.append('no')
            
            print(f"Frame {frame_count}: Predicted Class - {pred_class}")
        
    cap.release()
    print("Poaching detection completed.")
    print(f"Number of 'yes' frames: {len(yes_list)}")
    print(f"Number of 'no' frames: {len(no_list)}")
    selected_frames = random.sample(detected_frames, min(5, len(detected_frames)))
    
    poaching_detected = len(yes_list) > 0
    
    return {'poaching_detected': poaching_detected}, selected_frames


if __name__ == '__main__':
    video_path = 'sample-8.mp4'
    garuda_drishti_model=GarudaDrishti(num_classes=1,resnet_weights=resnet_big_weights).to(device)
    garuda_drishti_model.load_state_dict(torch.load('garuda_drishtiV1.pth'))
    result, selected_frames = detect_poaching_on_video(video_path, garuda_drishti_model, poach_transforms_, poaching_class_list, device)
    print(result)
    if len(selected_frames) ==0:
        print("No poaching detected in the selected frames.")
    else:
        fig, axes = plt.subplots(1, len(selected_frames), figsize=(20, 10))
        for i, frame in enumerate(selected_frames):
            axes[i].imshow(frame)
            axes[i].axis('off')
        plt.show()
    
    