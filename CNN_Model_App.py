import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

class CNN_Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.45),

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
    return CNN_Model(), device

def prep_img(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if image.mode != 'RGB':
        img_trans_ed = image.convert('RGB')
    
    return transform(img_trans_ed).unsqueeze(0)

def prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return predicted.item(), confidence.item()

def main():
    st.title("Car Damage Detection")    
    uploaded_file = st.file_uploader("Choose car image", type=['jpg', 'jpeg', 'png'])

    dmg_types = [
        'Door dent',
        'Bumper dent', 
        'Door scratch',
        'Bumper scratch',
        'Broken glass',
        'Broken tail light',
        'Broken head light'
    ]
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        model, device = get_model()
        
        if st.button("Analyze Damage"):
                image_tensor = prep_img(img)
                class_idx, confidence = prediction(model, image_tensor, device)
                
                damage = dmg_types[class_idx]
                confidence_pct = confidence * 100
                
                st.success("Analysis Complete!")
                st.metric("Detected Damage", damage, f"{confidence_pct:.1f}% confidence")
                
                if confidence_pct > 80:
                    st.success("High confidence in this detection")
                elif confidence_pct > 60:
                    st.info("Moderate confidence in this detection")
                else:
                    st.warning("Low confidence - may need manual verification")

if __name__ == "__main__":
    main()