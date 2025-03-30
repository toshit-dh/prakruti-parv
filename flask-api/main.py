import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask,request,jsonify,send_file,send_from_directory
from flask_cors import CORS
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torchvision.models as models
from PIL import Image,ImageDraw,ImageFont
import re
import cv2
import random
import base64
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn.functional as F

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
NEWS_API_KEY=os.getenv('NEWS_API_KEY')

if not api_key:
    raise ValueError("No API key found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)
genai_model= genai.GenerativeModel('gemini-1.5-flash')

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

resnet_big_weights=models.ResNet101_Weights.DEFAULT
class GarudaDrishti(nn.Module):
    def __init__(self,num_classes,resnet_weights):
        super().__init__()
        self.model=models.resnet101(weights=resnet_weights)
        self.model.fc=nn.Linear(in_features=self.model.fc.in_features,out_features=1)
        
    def forward(self,X:torch.Tensor):
        return self.model(X)
    
garuda_drishti_model=GarudaDrishti(num_classes=1,resnet_weights=resnet_big_weights).to(device)
garuda_drishti_model.load_state_dict(torch.load('garuda_drishtiV1.pth'))
poaching_class_list=['no','yes']
poach_transforms_=transforms.Compose([
    transforms.Resize((224, 224)),  
    #transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
                    image=transforms_(image).unsqueeze(0).to(device) # type: ignore
                    output=model(image)
                    pred_label=torch.argmax(torch.softmax(output,dim=1),dim=1)
                    pred_label=class_list[pred_label]
                    #print(pred_label)
                    response=genai_model.generate_content(f' Work as per {input_prompt}.{pred_label} is your given animal').text
                    #print(response)
                    kingdom = re.search(r'Kingdom:\s*(.*)', response).group(1) # type: ignore
                    phylum = re.search(r'Phylum:\s*(.*)', response).group(1)  # type: ignore
                    animal_class = re.search(r'Class:\s*(.*)', response).group(1) # type: ignore
                    order = re.search(r'Order:\s*(.*)', response).group(1)   # type: ignore
                    family = re.search(r'Family:\s*(.*)', response).group(1)   # type: ignore
                    scientific_name = re.search(r'Scientific Name:\s*(.*)', response).group(1)  # type: ignore
                    primary_habitat = re.search(r'Primary Habitat:\s*(.*)', response).group(1)   # type: ignore
                    geographical_range = re.search(r'Geographical Range:\s*(.*)', response).group(1)   # type: ignore
                    fun_fact = re.search(r'Fun Fact:\s*(.*)', response).group(1)   # type: ignore
                    legal_consequence_1 = re.search(r'Legal Consequence 1:\s*(.*)', response).group(1)   # type: ignore
                    legal_consequence_2 = re.search(r'Legal Consequence 2:\s*(.*)', response).group(1)   # type: ignore
                    past_crime_example_1 = re.search(r'Past Crime Example 1:\s*(.*)', response).group(1)  # type: ignore
                    past_crime_example_2 = re.search(r'Past Crime Example 2:\s*(.*)', response).group(1)  # type: ignore

                    
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
    

@app.route('/poach', methods=['POST', 'GET'])
def poach():
    if request.method == 'GET':
        return "<h1>Send a POST request with an video file to identify the animal.</h1>" 
    else:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided.'}), 400
        file = request.files['video']
        if file and file.filename != '':
            try:
                upload_folder = 'uploads'
                os.makedirs(upload_folder, exist_ok=True)
                video_path = os.path.join(upload_folder, file.filename) # type: ignore
                file.save(video_path)
                
                detection_result, detected_frames  = detect_poaching_on_video(
                    video_path=video_path,
                    model=garuda_drishti_model,
                    transform=poach_transforms_,
                    class_names=poaching_class_list,
                    device=device,
                    max_frames=300
                )
                
                os.remove(video_path)
                
                if detection_result['poaching_detected']: # type: ignore
                    encoded_frames = []
                    for frame in detected_frames:
                        buffered = BytesIO()
                        frame.save(buffered, format="JPEG") #type: ignore
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        encoded_frames.append(f"data:image/jpeg;base64,{img_str}")
                        
                    return jsonify({
                        'poaching_detected': True,
                        'details': 'Poaching activities were detected in the uploaded video.',
                        'detected_frames': encoded_frames
                    }), 200
                else:
                    return jsonify({
                        'poaching_detected': False,
                        'details': 'No poaching activities were detected in the uploaded video.'
                    }), 200
            except Exception as e:
                return jsonify({'error': f'An error occurred while processing the video: {str(e)}'}), 500
        else:
            return jsonify({'error': 'No video file found.'}), 400

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


def create_certificate(name):
    try:
        certificate_image = Image.open("template/prakruti-parv-cetificate-template.png").convert("RGBA")  

        draw = ImageDraw.Draw(certificate_image)       
        try:
            font = ImageFont.truetype("font/AlexBrush-Regular.ttf", 170) 
        except IOError:
            font = ImageFont.load_default()
            print("Custom font not found. Using default font.")
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

       
        text_position = (
            (certificate_image.width - text_width) // 2,
            550  
        )

        draw.text(text_position, name, font=font, fill="black")  
        img_io = BytesIO()
        certificate_image.save(img_io, 'PNG')
        img_io.seek(0)

        return img_io

    except Exception as e:
        print(f"Error creating certificate: {e}")
        return None
    
@app.route('/generate-certificate', methods=['GET'])
def certificate():
    name = request.args.get('name', None)
    if not name:
        return jsonify({'error': 'No name provided. Please provide a name query parameter.'}), 400

    certificate_image = create_certificate(name)
    if certificate_image:
        return send_file(
            certificate_image,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{name}_certificate.png"
        )
    else:
        return jsonify({'error': 'Failed to create certificate.'}), 500
    
    
def search_wildlife_videos(animal_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    url = f"https://www.youtube.com/results?search_query={animal_name}+wildlife"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        video_ids = re.findall(r'watch\?v=(\S{11})', response.text)
    
        video_links = list(set(f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids))
        
        return video_links
    
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return []

@app.route('/get-youtube-videos', methods=['GET'])
def get_youtube_videos():
    animal_name = request.args.get('name') 
    
    if not animal_name:
        return jsonify({"error": "Please provide an animal name in the 'name' query parameter"}), 400
    
    video_links = search_wildlife_videos(animal_name)
    
    if not video_links:
        return jsonify({"error": "No videos found or failed to fetch data"}), 404
    
    return jsonify({"animal": animal_name, "videos": video_links})


@app.route('/get-animal-sound', methods=['GET'])
def get_animal_sound():
    animal_name = request.args.get('animal')
    
    if not animal_name:
        return jsonify({'error': 'Animal name is required'}), 400
    
    animal_name = animal_name.lower()
    
    base_path = os.path.join(os.path.dirname(__file__), 'animal-sounds')
    animal_folder = os.path.join(base_path, animal_name)
    
    if not os.path.exists(animal_folder):
        return jsonify({'error': f'Sound not found for {animal_name}'}), 404
    
    audio_files = [f for f in os.listdir(animal_folder) 
                   if f.endswith(('.mp3', '.wav', '.ogg'))]
    
    if not audio_files:
        return jsonify({'error': f'No audio file found for {animal_name}'}), 404
    
    audio_file = audio_files[0]
    try:
        return send_from_directory(
            animal_folder,
            audio_file,
            as_attachment=True,
            download_name=f"{animal_name}_sound{os.path.splitext(audio_file)[1]}"
        )
    except Exception as e:
        return jsonify({'error': f'Error sending file: {str(e)}'}), 500


usernames = [
    "IndiAves",
    "indian_wildlife",
    "WildlifeMoment",
    "RanthamboreTig2",
    "PugdundeeSafari"
]

def fetch_recent_tweet_links(usernames):
    tweet_links = []

    for username in usernames:
        url = f'https://twitter.com/{username}'
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch {username}: {response.status_code}")
            continue  

        # Use regex to find all tweet links in the response text
        tweet_ids = re.findall(r'/status/(\d+)', response.text)
        print(response.text)
        return []
        for tweet_id in tweet_ids:
            tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
            tweet_links.append(tweet_url)

    return tweet_links


@app.route('/fetch_tweets', methods=['GET'])
def fetch_tweets():
    tweet_links = fetch_recent_tweet_links(usernames)
    print(tweet_links)
    
    return jsonify({"tweet_links": tweet_links})

@app.route('/fetch_indian_news', methods=['GET'])
def fetch_indian_news_view():
    url = "https://newsapi.org/v2/everything"
    keywords = [
        "wildlife conservation India",
        "poaching India",
        "endangered species India",
        "wildlife protection laws India",
        "Project Tiger India",
        "Project Elephant India",
        "wildlife sanctuaries India",
        "national parks India",
        "biodiversity India",
        "illegal wildlife trade India",
        "habitat loss India",
        "wildlife trafficking India",
        "forest conservation India",
        "community-based conservation India",
        "wildlife research India",
        "human-wildlife conflict India"
    ]

    query = " OR ".join(keywords)
    query="wildlife conservation India"
    
    params = {
        "q": query,
        "language": "en",  
        "sortBy": "relevancy",  
        "apiKey":  NEWS_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": response.json()}), response.status_code
    

@app.route('/fetch_default_tweets', methods=['GET'])
def fetch_default_tweets():
    tweets = [
        {
            "title": "Wildlife Conservation Efforts",
            "link": "https://twitframe.com/show?url=https://twitter.com/WildlifeSOS/status/1522206982454743040"
        },
        {
            "title": "Save Our Tigers",
            "link": "https://twitframe.com/show?url=https://twitter.com/IndianTechGuide/status/1838526292880429405"
        },
        {
            "title": "Endangered Species Awareness",
            "link": "https://twitframe.com/show?url=https://twitter.com/ColoursOfBharat/status/1588782226350931968"
        },
        {
            "title": "Project Tiger Updates",
            "link": "https://twitframe.com/show?url=https://twitter.com/PMOIndia/status/1343070549711319040"
        },
        {
            "title": "Wildlife Sanctuaries in India",
            "link": "https://twitframe.com/show?url=https://twitter.com/the_wildindia/status/1288390989384577032"
        },
        {
            "title": "Human-Wildlife Conflict Solutions",
            "link": "https://twitframe.com/show?url=https://twitter.com/Rainmaker1973/status/1798680635479126503"
        }
    ]
    
    return jsonify(tweets)
    
class Config:
    sample_rate = 44100  
    num_samples = 176400  
    n_mels = 128  
    n_fft = 2048  
    hop_length = 512  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 13  

class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512):
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = AmplitudeToDB()
    
    def __call__(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        return mel_spec_db

class AudioCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)
        
      
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc_input_size = 256 * 8 * 21
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
    
        x = x.reshape(x.size(0), -1)
        x = self.dropout5(F.relu(self.bn5(self.fc1(x))))
        x = self.fc2(x)
        return x

def load_class_mapping(data_path):
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    class_mapping = {idx: class_name for idx, class_name in enumerate(sorted(class_dirs))}
    return class_mapping

def predict_animal_sound(audio_file_path, model_path, data_path):

    cfg = Config()
    class_mapping = load_class_mapping(data_path)
    
    feature_extractor = AudioFeatureExtractor(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length
    )
    
    model = AudioCNN(num_classes=cfg.num_classes).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
  
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, cfg.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[1] < cfg.num_samples:
            padding = cfg.num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > cfg.num_samples:
            waveform = waveform[:, :cfg.num_samples]
        features = feature_extractor(waveform).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            outputs = model(features)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            predicted_animal = class_mapping[predicted_idx] # type: ignore
            confidence = probabilities[predicted_idx].item() * 100 # type: ignore
            top3_values, top3_indices = torch.topk(probabilities, 3)
            top3_predictions = [
                (class_mapping[idx.item()], prob.item() * 100) # type: ignore
                for idx, prob in zip(top3_indices, top3_values)
            ]
                   
        return predicted_animal, confidence, top3_predictions
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None, 0, []
    
    
@app.route('/predict-sound', methods=['POST', 'GET'])
def predict_sound():
    if request.method == 'GET':
        return "<h1>Send a POST request with an audio file to identify the animal sound.</h1>"
    else:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided.'}), 400
        
        file = request.files['audio']
        if file and file.filename != '':
            try:
                upload_folder = 'temp_audio'
                os.makedirs(upload_folder, exist_ok=True)
                audio_path = os.path.join(upload_folder, file.filename) # type: ignore
                file.save(audio_path)
                
                model_path = 'best_animal_sound_model.pth'  
                data_path = 'Animal-Sound'  
                
                predicted_animal, confidence, top3_predictions = predict_animal_sound(
                    audio_path, model_path, data_path
                )
                
                os.remove(audio_path)
                
                if predicted_animal:
                    return jsonify({
                        'animal': predicted_animal,
                        'confidence': confidence,
                        'top_predictions': top3_predictions
                    }), 200
                else:
                    return jsonify({'error': 'Could not identify the animal sound.'}), 400
                    
            except Exception as e:
                return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500
        else:
            return jsonify({'error': 'No audio file found.'}), 400

if __name__ == '__main__':
    app.run(port=8081, debug=True)