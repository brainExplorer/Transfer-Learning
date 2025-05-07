Transfer Learning with MobileNetV2 on Flowers102
This project demonstrates transfer learning using MobileNetV2 for classifying flower images from the Flowers102 dataset. The model is fine-tuned for 102 flower species and evaluated with a confusion matrix and classification report.
Features
✅ Uses MobileNetV2 pretrained on ImageNet
🌸 Fine-tuned for 102 flower classes
🔁 Data augmentation on training data
📊 Evaluation with confusion matrix and classification report
💾 Saves the trained model weights
Model Architecture
Base model: torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
Classifier: Replaced last linear layer with nn.Linear(num_ftrs, 102)
Dataset
Source: Flowers102 Dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
Accessed via: torchvision.datasets.Flowers102
Splits used: train and val
Setup
1. Clone the Repository
git clone https://github.com/yourusername/flowers102-mobilenetv2.git
cd flowers102-mobilenetv2
2. Install Dependencies
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
Run the Code
python main.py
Evaluation Outputs
✅ Random sample visualization from training set
📉 Epoch-wise training loss printed to console
🔷 Confusion matrix plotted using seaborn
📄 Classification report printed with precision, recall, and F1-score
Sample Results
Training loss: ~0.5 – 0.7
Validation accuracy: ~75% – 85%
Saved model file: mobilenetv2_flowers102.pth
Configurable Parameters
You can adjust the following in the script:
epochs = 3
batch_size = 32
learning_rate = 0.001
step_size = 5  # scheduler step size
gamma = 0.1    # scheduler decay factor
To fine-tune the whole model (not just the classifier), you can unfreeze all layers:

for param in model.features.parameters():
    param.requires_grad = True
Saving and Loading the Model
Saving: torch.save(model.state_dict(), 'mobilenetv2_flowers102.pth')
Loading:
model.load_state_dict(torch.load('mobilenetv2_flowers102.pth'))
model.eval()
Notes
- Data augmentation is applied only to training data.
- Test data is only resized and normalized.
- Pretrained weights come from ImageNet1K.
License
This project is licensed under the MIT License. See the LICENSE file for full license text.
Acknowledgements
- Oxford VGG Group for the Flowers102 dataset
- PyTorch, Torchvision
- Seaborn for plotting
- Scikit-learn for metrics
Author
Developed by Ümit YAVUZ
