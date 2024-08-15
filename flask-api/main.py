import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from PIL import Image
import re


load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)
genai_model= genai.GenerativeModel('gemini-pro')

app = Flask(__name__)
CORS(app)

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
model=WildResNet(num_classes=len(class_list),resnet_weights=resnet_weights).to(device)
print(f"Model is running on :{next(model.parameters()).device}")
model.load_state_dict(torch.load('wildResNet.pth'))

input_prompt = """
Act as an animal knowledge expert. Given the name of an animal, provide the following information in a structured format. Strictly follow this format for the response:

Scientific Classification:
1. Kingdom: Provide the biological kingdom of the animal (e.g., Animalia).
2. Phylum: Specify the phylum (e.g., Chordata).
3. Class: Identify the class to which the animal belongs (e.g., Mammalia).
4. Order: State the biological order (e.g., Carnivora).
5. Family: Specify the family (e.g., Felidae).
6. Scientific Name: Provide the scientific name (e.g., Panthera tigris).

Location:
7. Primary Habitat: Describe where this animal is mostly found (e.g., tropical forests, savannas).
8. Geographical Range: Mention specific countries or regions where this animal is found.

Interesting Points:
9. Fun Fact: Share a fun or lesser-known fact about this animal.

Legal Consequences of Harm:
10. Legal Consequence 1: Describe a specific crime that applies if someone kills or harms this animal. Include the relevant law or act under which this crime is punishable.
11. Legal Consequence 2: Describe another specific crime related to harming this animal. Include the relevant law or act under which this crime is punishable.

Past Crime Examples:
12. Past Crime Example 1: Provide a past example of any country where individuals were prosecuted or penalized for killing or harming this animal by law. If the year is available, include it in the response .If year is not available,leave it ..dont include it in repsonse by doing[Year]It looks bad.
13. Past Crime Example 2: Provide another past example of any country where individuals were prosecuted or penalized for killing or harming this animal by law. If the year is available, include it in the response.If year is not available,leave it ..dont include it in repsonse by doing[Year]It looks bad.
"""


@app.route('/',methods= ['GET'])
def welcome():
    return "<h1>Welcome to Wild Animal Identification API</h1>"


@app.route('/identify',methods=['POST','GET'])
def identify():
    if request.method == 'GET':
        return "<h1>Send a POST request with an image file to identify the animal.</h1>"
    else:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400
        
        file = request.files['image']
        if file and file.filename != '':
            
            try:
                image = Image.open(file.stream)
                model.eval()
                with torch.inference_mode():
                    image=transforms_(image).unsqueeze(0).to(device)
                    output=model(image)
                    pred_label=torch.argmax(torch.softmax(output,dim=1),dim=1)
                    pred_label=class_list[pred_label]
                    #print(pred_label)
                    response=genai_model.generate_content(f' Work as per {input_prompt}.{pred_label} is your given animal').text
                    #print(response)
                    kingdom = re.search(r'Kingdom:\s*(.*)', response).group(1)
                    phylum = re.search(r'Phylum:\s*(.*)', response).group(1)
                    animal_class = re.search(r'Class:\s*(.*)', response).group(1)
                    order = re.search(r'Order:\s*(.*)', response).group(1)
                    family = re.search(r'Family:\s*(.*)', response).group(1)
                    scientific_name = re.search(r'Scientific Name:\s*(.*)', response).group(1)
                    primary_habitat = re.search(r'Primary Habitat:\s*(.*)', response).group(1)
                    geographical_range = re.search(r'Geographical Range:\s*(.*)', response).group(1)
                    fun_fact = re.search(r'Fun Fact:\s*(.*)', response).group(1)
                    legal_consequence_1 = re.search(r'Legal Consequence 1:\s*(.*)', response).group(1)
                    legal_consequence_2 = re.search(r'Legal Consequence 2:\s*(.*)', response).group(1)
                    past_crime_example_1 = re.search(r'Past Crime Example 1:\s*(.*)', response).group(1)
                    past_crime_example_2 = re.search(r'Past Crime Example 2:\s*(.*)', response).group(1)

                    
                    return jsonify({
                        'label': pred_label,
                        'kingdom': kingdom,
                        'phylum': phylum,
                        'class': animal_class,
                        'order': order,
                        'family': family,
                        'scientific_name': scientific_name,
                        'primary_habitat': primary_habitat,
                        'geographical_range': geographical_range,
                        'fun_fact': fun_fact,
                        'crime_1': legal_consequence_1,
                        'crime_2': legal_consequence_2,
                        'example_1': past_crime_example_1,
                        'example_2': past_crime_example_2
                    }), 200
            except IOError:
                return jsonify({'error': 'Invalid image file.'}), 400
        else:
            return jsonify({'error': 'No image file found.'}), 400
    
    
    
if __name__ == '__main__':
    app.run(port=8081, debug=True)