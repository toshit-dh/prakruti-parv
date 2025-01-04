import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from PIL import Image,ImageDraw,ImageFont


transforms_=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    
])
resnet_weights=models.ResNet50_Weights.DEFAULT
class WildResNet(nn.Module):
    def __init__(self,num_classes,resnet_weights):
        super().__init__()
        self.model=models.resnet50(weights=resnet_weights)
        self.model.fc=nn.Linear(in_features=self.model.fc.in_features,out_features=num_classes)
        
    def forward(self,X:torch.Tensor):
        return self.model(X)
    
class_list=['Aardwolf', 'Alpaca', 'Armadillo', 'Baboon', 'Bighorn Sheep', 'Binturong', 'Bobcat', 'Camel', 'Capybara', 'Caracal', 'Cheetah', 'Cougar', 'Dugong', 'Eland', 'Fennec Fox', 'Galago (Bushbaby)', 'Gaur', 'Gerenuk', 'Gibbon', 'Giraffe', 'Grizzly Bear', 'Honey Badger', 'Ibex', 'Jackal', 'Jaguar', 'Jerboa', 'King Cobra', 'Kinkajou', 'Klipspringer', 'Kudu', 'Lemur', 'Lion-tailed macaque', 'Llama', 'Lynx', 'Margay', 'Meerkat', 'Mongoose', 'Mouse deer', 'Mule Deer', 'Musk Ox', 'Nilgai', 'Nilgiri Langur', 'Nilgiri_tahr', 'Nyala', 'Pangolin', 'Patagonian Mara', 'Peacock', 'Peccary', 'Platypus', 'Polar Bear', 'Pronghorn', 'Puma', 'Quokka', 'Quoll', 'Red Panda', 'Ring-tailed Lemur', 'Rock Hyrax', 'Sloath_bear', 'Sloth', 'Snow Leopard', 'Squirrel Monkey', 'Striped Hyena', 'Takin', 'Tapir', 'Tasmanian Devil', 'Tenrec', 'Toad', 'Tortoise', 'Vicuna', 'Warthog', 'Wolverine', 'Xenoceratops', 'Yak', 'aardavark', 'ant', 'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chameleon', 'chimpanzee', 'cow', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hen', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'pigeon', 'porcupine', 'rhinoceros', 'seahorse', 'seal', 'shark', 'sheep', 'snail', 'snake', 'sparrow', 'squirrel', 'swan', 'tiger', 'turkey', 'turtle', 'ukari', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']        

device="cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(imag_path):
    model=WildResNet(num_classes=len(class_list),resnet_weights=resnet_weights).to(device)
    print(f"Model is running on :{next(model.parameters()).device}")
    model.load_state_dict(torch.load('wildResNet.pth'))
    img=Image.open(imag_path)
    img=transforms_(img).unsqueeze(0)
    return img.to(device),model

def predict_animal(image_tensor,model):
    model.eval()
    with torch.inference_mode():
        output=model(image_tensor)
        pred_label=torch.argmax(torch.softmax(output,dim=1),dim=1)
        pred_label=class_list[pred_label]
    return pred_label

if __name__=="__main__":
    image_path="animal-img.jpg"
    image_tensor,model=preprocess_image(image_path)
    pred_label=predict_animal(image_tensor,model)
    print(f"Predicted Animal is :{pred_label}")
