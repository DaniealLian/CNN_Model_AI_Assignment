import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Your model architecture (same as before)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN, self).__init__()
        
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model(model_path, num_classes=7):
    """
    Load the trained model. Cached to avoid reloading on every interaction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(num_classes=num_classes)
    
    try:
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success(f"Model loaded successfully on {device}")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction.
    """
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standards
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_tensor, device, class_names):
    """
    Make prediction on the preprocessed image.
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()

def create_probability_chart(probabilities, class_names):
    """
    Create a bar chart showing prediction probabilities for all classes.
    """
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities
    })
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df, 
        x='Probability', 
        y='Class',
        orientation='h',
        title='Prediction Probabilities by Class',
        color='Probability',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Class",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Custom CNN Image Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🖼️ Custom CNN Image Classifier")
    st.markdown("Upload an image to get predictions from your trained CNN model!")
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model settings
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="model_weights.pth", 
        help="Path to your trained model weights (.pth file)"
    )
    
    num_classes = st.sidebar.number_input(
        "Number of Classes", 
        min_value=1, 
        max_value=50, 
        value=7, 
        help="Number of classes your model was trained on"
    )
    
    # Class names - you should replace these with your actual class names
    default_classes = [f"Class_{i}" for i in range(num_classes)]
    class_names = st.sidebar.text_area(
        "Class Names (one per line)",
        value="\n".join(default_classes),
        help="Enter the names of your classes, one per line"
    ).split('\n')
    
    # Ensure class names match number of classes
    if len(class_names) != num_classes:
        st.sidebar.warning(f"Number of class names ({len(class_names)}) doesn't match number of classes ({num_classes})")
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Image preprocessing settings
    st.sidebar.subheader("Image Preprocessing")
    target_width = st.sidebar.number_input("Target Width", min_value=32, max_value=512, value=224)
    target_height = st.sidebar.number_input("Target Height", min_value=32, max_value=512, value=224)
    
    # Load model
    try:
        model, device = load_model(model_path, num_classes)
    except:
        st.error("Please ensure your model file is in the correct location and try again.")
        st.stop()
    
    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image file for classification"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Preprocess the image
                with st.spinner("Processing image..."):
                    image_tensor = preprocess_image(image, (target_height, target_width))
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    predicted_class, confidence, probabilities = predict(
                        model, image_tensor, device, class_names
                    )
                
                # Display results
                st.success("Prediction completed!")
                
                # Main prediction
                st.metric(
                    label="Predicted Class",
                    value=class_names[predicted_class],
                    delta=f"{confidence*100:.2f}% confidence"
                )
                
                # Confidence indicator
                if confidence > 0.8:
                    st.success(f"High confidence prediction!")
                elif confidence > 0.6:
                    st.warning(f"Medium confidence prediction")
                else:
                    st.error(f"Low confidence prediction")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.stop()
    
    # Probability visualization
    if uploaded_file is not None and 'probabilities' in locals():
        st.subheader("📊 Detailed Predictions")
        
        # Create probability chart
        fig = create_probability_chart(probabilities, class_names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability table
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': probabilities,
            'Percentage': [f"{p*100:.2f}%" for p in probabilities]
        }).sort_values('Probability', ascending=False)
        
        st.dataframe(prob_df, use_container_width=True)
    
    # Model information
    st.sidebar.subheader("ℹ️ Model Information")
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.sidebar.write(f"**Total Parameters:** {total_params:,}")
        st.sidebar.write(f"**Trainable Parameters:** {trainable_params:,}")
        st.sidebar.write(f"**Device:** {device}")
    
    # About section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This Streamlit app uses your custom CNN model to classify images into one of the trained categories.
        
        **How to use:**
        1. Configure the model path and class names in the sidebar
        2. Upload an image using the file uploader
        3. View the prediction results and confidence scores
        
        **Model Architecture:**
        - 5 convolutional layers with batch normalization and LeakyReLU activation
        - Global average pooling
        - 3 fully connected layers with dropout for regularization
        """)

if __name__ == "__main__":
    main()