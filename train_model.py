from fastai.vision.all import *
import os
from fastai.callback.progress import CSVLogger
import torch.nn.functional as F

def get_data():
    # Path to the MNIST data directory
    data_path = Path('../mnist_data')
    
    # Create ImageDataLoaders with enhanced transformations
    dls = ImageDataLoaders.from_folder(
        data_path,
        valid_pct=0.2,
        item_tfms=[ToTensor(), Resize(28)],
        batch_tfms=[
            *aug_transforms(
                do_flip=False,  # No flipping for digits
                max_rotate=15.0,  # Slightly more rotation
                max_zoom=1.1,
                max_lighting=0.2,
                max_warp=0.2,
                p_affine=0.7,  # Probability of applying affine transforms
                p_lighting=0.7  # Probability of applying lighting transforms
            ),
            Normalize.from_stats(*imagenet_stats)
        ],
        num_workers=0,
        bs=64  # Reduced batch size for stability
    )
    return dls

# Enhanced CNN architecture with residual connections
class MNISTResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        # Residual blocks
        self.res1 = self._make_res_block(32, 64)
        self.res2 = self._make_res_block(64, 128)
        self.res3 = self._make_res_block(128, 256)
        
        # Final layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 10)
        
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Global average pooling and final layers
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model():
    print("Starting model training...")
    
    # Get data
    dls = get_data()
    
    # Create model and learner
    model = MNISTResNet()
    learn = Learner(
        dls,
        model,
        loss_func=F.cross_entropy,
        metrics=[accuracy, error_rate],
        cbs=[
            CSVLogger(),
            SaveModelCallback(monitor='accuracy', comp=np.greater, fname='best_model'),
            ReduceLROnPlateau(monitor='valid_loss', patience=2)
        ]
    )
    
    # Find optimal learning rate
    print("\nFinding optimal learning rate...")
    suggested_lr = learn.lr_find().valley
    
    # Initial training phase
    print(f"\nTraining the model with learning rate: {suggested_lr}")
    learn.fit_one_cycle(
        20,  # Initial training epochs
        suggested_lr
    )
    
    # Fine-tuning phase
    print("\nFine-tuning the model...")
    learn.fit_one_cycle(
        10,  # Fine-tuning epochs
        suggested_lr/10
    )
    
    # Load best model
    learn.load('best_model')
    
    # Save the model state dictionary
    model_path = Path('../mnist_data/mnist_model.pkl')
    torch.save(learn.model.state_dict(), model_path)
    print(f"\nBest model weights saved to {model_path}")
    
    # Print final metrics
    print("\nFinal metrics:")
    print(f"Validation Accuracy: {learn.validate()[1]:.4f}")
    print(f"Validation Error Rate: {learn.validate()[2]:.4f}")

if __name__ == "__main__":
    train_model() 