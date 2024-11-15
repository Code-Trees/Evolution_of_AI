import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Trainer:
    def __init__(self, training_state):
        self.training_state = training_state
        
        # Initialize model and device
        self.model = Net()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize training parameters
        self.num_epochs = 20
        self.learning_rate = 0.01
        self.batch_size = 64
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics storage
        self.training_losses1 = []
        self.training_losses2 = []
        self.training_accuracies1 = []
        self.training_accuracies2 = []
        self.start_epoch = 0
        
        # Initialize data loaders
        self.train_loader = None
        self.test_loader = None
        self.setup_data_loaders()

    def setup_data_loaders(self):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def configure_training(self, config):
        """Configure training parameters from request data"""
        try:
            # Update training parameters
            self.num_epochs = config.get('num_epochs', 20)
            self.learning_rate = config.get('learning_rate', 0.01)
            self.batch_size = config.get('batch_size', 64)
            self.start_epoch = config.get('start_epoch', 0)
            
            # Configure optimizer
            optimizer_name = config.get('optimizer', 'SGD')
            if optimizer_name == 'SGD':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Update model architecture if provided
            model_configs = config.get('model_configs', {})
            if model_configs:
                self.update_model_architecture(model_configs)
            
            # Recreate data loaders with new batch size if it changed
            self.setup_data_loaders()
            
            print(f"Training configured with: epochs={self.num_epochs}, lr={self.learning_rate}, batch_size={self.batch_size}")
            
        except Exception as e:
            print(f"Error in configure_training: {str(e)}")
            raise

    def update_model_architecture(self, configs):
        """Update model architecture based on configurations"""
        model_config = configs.get('model1', {})  # Get model1 configs
        self.model = Net(
            conv1_channels=model_config.get('conv1', 16),
            conv2_channels=model_config.get('conv2', 32),
            conv3_channels=model_config.get('conv3', 64)
        ).to(self.device)

    def train_model(self):
        try:
            print(f"Starting training for {self.num_epochs} epochs")
            for epoch in range(self.start_epoch, self.num_epochs):
                if not self.training_state.is_training:
                    print("Training stopped")
                    break
                
                # Training loop here
                self.model.train()
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if not self.training_state.is_training:
                        break
                        
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    # Store metrics (simplified)
                    if batch_idx % 100 == 0:
                        self.training_losses1.append({
                            'epoch': epoch,
                            'train_loss': loss.item(),
                            'test_loss': 0.0  # Add proper test loss calculation if needed
                        })
                
                self.training_state.current_epoch = epoch + 1
                print(f"Completed epoch {epoch + 1}")
                
        except Exception as e:
            print(f"Error in training: {str(e)}")
        finally:
            self.training_state.is_training = False

    def reset_model(self):
        """Reset the model to initial state"""
        try:
            self.model = Net().to(self.device)
            self.optimizer = None
            print("Model reset successfully")
        except Exception as e:
            print(f"Error in reset_model: {str(e)}")
            raise

    def reset_metrics(self):
        """Reset all training metrics"""
        self.training_losses1 = []
        self.training_losses2 = []
        self.training_accuracies1 = []
        self.training_accuracies2 = []
        self.start_epoch = 0
        print("Metrics reset successfully")

    def get_metrics(self):
        """Get current training metrics"""
        return {
            'model1': {
                'losses': self.training_losses1,
                'accuracies': self.training_accuracies1
            }
        }
    