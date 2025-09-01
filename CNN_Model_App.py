import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

class CNN_Model(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN_Model, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4)
        )
        
        # Global Average Pooling and Fully Connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.45),

            # Dense
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),

            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_Model()
    
    # Load pre-trained weights - you need to have a trained model file
    # Replace 'model_weights.pth' with your actual model file path
    model_path = 'model_weights.pth'  # Update this path to your trained model file
    
    if os.path.exists(model_path):
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.warning(f"Model weights file not found at {model_path}. Using untrained model.")
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

def prep_img(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standards
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return predicted.item(), confidence.item()

def main():
    st.title("Car Damage Detection")    
    img_input = st.file_uploader("Choose car image", type=['jpg', 'jpeg', 'png'])

    dmg_types = [
        'Door dent',
        'Bumper dent', 
        'Door scratch',
        'Bumper scratch',
        'Broken glass',
        'Broken tail light',
        'Broken head light'
    ]
    
    if img_input:
        image = Image.open(img_input)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add a loading spinner while processing
        with st.spinner('Loading model and analyzing image...'):
            model, device = get_model()
            image_tensor = prep_img(image)
            predict_result, confidence = prediction(model, image_tensor, device)
        
        damage = dmg_types[predict_result]
        conf_percent = confidence * 100
        
        st.success("Analysis Complete!")
        st.metric(
            label="Detected Damage",
            value=damage, 
            delta=f"{conf_percent:.1f}% confidence"
        )
        
        if conf_percent > 80:
            st.success("High confidence in this detection")
        elif conf_percent > 60:
            st.info("Moderate confidence in this detection")
        else:
            st.warning("Low confidence - may need manual verification")

if __name__ == "__main__":
    main()