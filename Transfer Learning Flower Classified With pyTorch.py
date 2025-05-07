"""
Transfer learning: Classification of flowers with 102 different species labels using a pre-trained model.
Transfer learing with MobileNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # %% data loading and data augmentation

    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),   # MobileNetV2 input size
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
        transforms.RandomRotation(10),    # Randomly rotate the image by 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change the brightness, contrast, saturation and hue of the image
        transforms.ToTensor(), # Convert the image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # You couldn't augment the test data, so you just resize and normalize it
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),   # MobileNetV2 input size
        transforms.ToTensor(), # Convert the image to a tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.Flowers102(
        root='./data',  # Path to the dataset
        split='train',           # Use the training set
        transform=transform_train, # Apply the transformations to the training set
        download=True            # Download the dataset if not already present
    )

    test_dataset = datasets.Flowers102(
        root='./data',  # Path to the dataset
        split='val',            # Use the test set
        transform=transform_test, # Apply the transformations to the test set
        download=True            # Download the dataset if not already present
    )

    # Select the random 5 samples
    indices = torch.randint(len(train_dataset), (5,))
    samples = [train_dataset[i] for i in indices]

    # visualize the samples
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, (image, label) in enumerate(samples):
        image = image.numpy().transpose((1, 2, 0))  # Convert to HWC format for visualization
        image = (image * 0.5 + 0.5)  # Unnormalize the image
        axes[i].imshow(image)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Batch size for training
        shuffle=True,   # Shuffle the data at every epoch
        num_workers=4,  # Number of workers for data loading
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Batch size for testing
        shuffle=False,  # No need to shuffle the test data
        num_workers=4,  # Number of workers for data loading
    )

    # %% transfwer learning definition and fine tuning and saving the model

    # mobilenetv2 loading - updated to use weights instead of pretrained
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # add classifier layer to the model
    num_ftrs = model.classifier[1].in_features  # Get the number of input features for the classifier layer
    model.classifier[1] = nn.Linear(num_ftrs, 102)  # Change the output layer to match the number of classes (102)  
    model = model.to(device)  # Move the model to the device (GPU or CPU)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)  # Optimizer for the classifier layer
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

    # training the model
    epochs = 3  # Number of epochs to train the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize the running loss

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):  # Iterate over the training data
            images, labels = images.to(device), labels.to(device)  # Move the data to the device

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item() * images.size(0)  # Update the running loss

        epoch_loss = running_loss / len(train_loader.dataset)  # Calculate the average loss for the epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')  # Print the loss for the epoch

        schedular.step()  # Step the learning rate scheduler
    
    # save the model
    torch.save(model.state_dict(), 'mobilenetv2_flowers102.pth')  # Save the model weights

    # %% test and evaluate the model
    model.eval()  # Set the model to evaluation mode
    all_predictions = []  # List to store all predictions
    all_labels = []  # List to store all labels
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)  # Move the data to the device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            all_predictions.extend(predicted.cpu().numpy())  # Store the predictions
            all_labels.extend(labels.cpu().numpy())  # Store the true labels    
    
    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)  # Calculate the confusion matrix
    plt.figure(figsize=(12,12))  # Set the figure size
    sns.heatmap(cm, annot=False, cmap='Blues')   
    # Plot the confusion matrix
    plt.xlabel('Predicted Label')  # Set the x-axis label
    plt.ylabel('True Label')  # Set the y-axis label
    plt.title('Confusion Matrix')  # Set the title
    plt.show()  # Show the plot
    print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))  # Print the classification report
    
    # Add your evaluation code here

if __name__ == '__main__':
    main()