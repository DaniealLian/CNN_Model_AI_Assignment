import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms


class BasicCnnModel(nn.Module):
    def __init__(self, num_classes=7):
        super(BasicCnnModel, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

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
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CnnDmgDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car body damage detector")
        self.root.geometry("500x600")
        self.root.configure(bg='white')
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image = None
        
        self.damage_types = [
            'Door dent',
            'Bumper dent', 
            'Door scratch',
            'Bumper scratch',
            'Broken glass',
            'Broken tail light',
            'Broken head light'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title = ttk.Label(main_frame, text="Car damage detector", font=('Arial', 16, 'bold'))
        title.pack(pady=(0, 20))

        select_btn = ttk.Button(main_frame, text="Select car image", command=self.insert_img, width=20)
        select_btn.pack(pady=10)
        
        self.selected_img = ttk.Label(main_frame, text="No image selected", font=('Arial', 10), foreground='gray')
        self.selected_img.pack(pady=20)
        
        self.analyse_dmg_btn = ttk.Button(main_frame, text="Analyse damage", command=self.analyse, state='disabled', width=20)
        self.analyse_dmg_btn.pack(pady=10)
        
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="15")
        results_frame.pack(fill=tk.X, pady=20)
        
        ttk.Label(results_frame, text="Detected damage:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.damage = tk.StringVar(value="No analysis yet")
        analysed_dmg_label = ttk.Label(results_frame, textvariable=self.damage, 
                                font=('Arial', 12), foreground='blue')
        analysed_dmg_label.pack(anchor=tk.W, pady=(5, 15))
        
        ttk.Label(results_frame, text="Confidence level:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.analysed_confidence = tk.StringVar(value="0%")
        analysed_conf_label = ttk.Label(results_frame, textvariable=self.analysed_confidence, 
                                    font=('Arial', 12), foreground='green')
        analysed_conf_label.pack(anchor=tk.W)
        
    def load_model(self):
        model_path = "cnn_model_weights.pth"
        
        self.model = BasicCnnModel(num_classes=7)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
    
    def insert_img(self):
        file_path = filedialog.askopenfilename(
            title="Select car image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            self.current_image = Image.open(file_path)
            
            display_image = self.current_image.copy()
            display_image.thumbnail((300, 200), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(display_image)
            self.selected_img.configure(image=photo, text="")
            self.selected_img.image = photo
            
            if self.model is not None:
                self.analyse_dmg_btn.configure(state='normal')
            
            self.damage.set("No analysis yet")
            self.analysed_confidence.set("0%")
    
    def analyse(self):
        self.root.update()
        
        image = self.current_image.convert('RGB')
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        dmg_type = self.damage_types[predicted_class]
        conf_percent = confidence_score * 100
        
        self.damage.set(dmg_type)
        self.analysed_confidence.set(f"{conf_percent:.2f}%")            

def main():
    root = tk.Tk()
    CnnDmgDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()