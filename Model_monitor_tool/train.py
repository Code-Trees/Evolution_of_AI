import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm
import base64
import io
from PIL import Image
import numpy as np
import os
from data_transform import get_data_transform
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import time

class TrainingState:
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0

class Trainer:
    def __init__(self, data_transform):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = None
        self.model2 = None
        self.optimizer1 = None
        self.optimizer2 = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = None
        self.test_loader = None
        self.best_accuracy1 = 0
        self.best_accuracy2 = 0
        self.batch_size = 64
        self.data_transform = data_transform
        
        # Initialize training state
        self.training_state = TrainingState()
        
        # Initialize logs
        self.logs = []
        
        # Initialize metrics storage
        self.training_losses1 = []
        self.training_losses2 = []
        self.training_accuracies1 = []
        self.training_accuracies2 = []
        
        # Create models directory
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Separate storage for epoch metrics and iteration losses
        self.epoch_metrics = {
            'model1': {
                'train_acc': [],
                'val_acc': [],
                'epochs': []
            },
            'model2': {
                'train_acc': [],
                'val_acc': [],
                'epochs': []
            }
        }
        
        self.iteration_losses = {
            'model1': [],
            'model2': []
        }
        
        # Initialize batch-wise loss tracking
        self.batch_metrics = {
            'model1': {
                'train_loss': [],
                'test_loss': [],
                'batch_numbers': []
            },
            'model2': {
                'train_loss': [],
                'test_loss': [],
                'batch_numbers': []
            }
        }
        self.current_batch = 0

    def setup_data_loaders(self):
        """Set up data loaders for training and testing"""
        try:
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Load MNIST dataset
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('data', train=False, transform=transform)
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            print("Data loaders initialized successfully")
            
        except Exception as e:
            print(f"Error setting up data loaders: {str(e)}")
            raise

    def start_training(self, num_epochs=20, learning_rate=0.01, batch_size=128, optimizer_name='SGD', model_configs=None):
        """Initialize and start the training process"""
        try:
            if self.training_state.is_training:
                print("Training already in progress")
                return False
                
            print(f"Initializing training with {num_epochs} epochs")
            
            # Set batch size
            self.batch_size = batch_size
            
            # Initialize models
            if model_configs:
                self.model1 = Net(
                    conv1_channels=model_configs['model1']['conv1'],
                    conv2_channels=model_configs['model1']['conv2'],
                    conv3_channels=model_configs['model1']['conv3']
                ).to(self.device)
                
                self.model2 = Net(
                    conv1_channels=model_configs['model2']['conv1'],
                    conv2_channels=model_configs['model2']['conv2'],
                    conv3_channels=model_configs['model2']['conv3']
                ).to(self.device)
            else:
                self.model1 = Net().to(self.device)
                self.model2 = Net().to(self.device)

            # Set up optimizers
            if optimizer_name == 'Adam':
                self.optimizer1 = optim.Adam(self.model1.parameters(), lr=learning_rate)
                self.optimizer2 = optim.Adam(self.model2.parameters(), lr=learning_rate)
            else:  # default to SGD
                self.optimizer1 = optim.SGD(self.model1.parameters(), lr=learning_rate)
                self.optimizer2 = optim.SGD(self.model2.parameters(), lr=learning_rate)

            # Set up training parameters
            self.training_state.total_epochs = num_epochs
            self.training_state.current_epoch = 0

            # Reset metrics
            self.training_losses1 = []
            self.training_losses2 = []
            self.training_accuracies1 = []
            self.training_accuracies2 = []

            # Initialize data loaders
            self.setup_data_loaders()
            
            print("Training initialization complete")
            return True

        except Exception as e:
            print(f"Error in start_training: {str(e)}")
            self.training_state.is_training = False
            return False

    def stop_training(self):
        """Stop the training process"""
        self.training_state.is_training = False

    def train_model(self):
        """Main training loop"""
        try:
            self.add_log(f"Starting training for {self.training_state.total_epochs} epochs")
            self.training_state.is_training = True
            self.current_batch = 0  # Reset batch counter
            
            for epoch in range(self.training_state.current_epoch, self.training_state.total_epochs):
                if not self.training_state.is_training:
                    break
                
                # Initialize metrics for this epoch
                total_correct1 = 0
                total_correct2 = 0
                total_samples = 0
                
                self.model1.train()
                self.model2.train()
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if not self.training_state.is_training:
                        break
                        
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = target.size(0)  # Get current batch size
                    
                    # Train and get losses for model1
                    train_loss1 = self.train_batch(self.model1, self.optimizer1, data, target)
                    test_loss1 = self.get_test_batch_loss(self.model1)
                    
                    # Train and get losses for model2
                    train_loss2 = self.train_batch(self.model2, self.optimizer2, data, target)
                    test_loss2 = self.get_test_batch_loss(self.model2)
                    
                    # Calculate accuracy for this batch
                    with torch.no_grad():
                        # Model 1 accuracy
                        output1 = self.model1(data)
                        pred1 = output1.argmax(dim=1, keepdim=True)
                        total_correct1 += pred1.eq(target.view_as(pred1)).sum().item()
                        
                        # Model 2 accuracy
                        output2 = self.model2(data)
                        pred2 = output2.argmax(dim=1, keepdim=True)
                        total_correct2 += pred2.eq(target.view_as(pred2)).sum().item()
                        
                        total_samples += batch_size
                    
                    # Store batch metrics
                    self.batch_metrics['model1']['batch_numbers'].append(self.current_batch)
                    self.batch_metrics['model1']['train_loss'].append(train_loss1)
                    self.batch_metrics['model1']['test_loss'].append(test_loss1)
                    
                    self.batch_metrics['model2']['batch_numbers'].append(self.current_batch)
                    self.batch_metrics['model2']['train_loss'].append(train_loss2)
                    self.batch_metrics['model2']['test_loss'].append(test_loss2)
                    
                    self.current_batch += 1
                    
                    if batch_idx % 10 == 0:  # Log every 10 batches
                        self.add_log(
                            f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                            f'Model1 Train/Test Loss: {train_loss1:.4f}/{test_loss1:.4f}, '
                            f'Model2 Train/Test Loss: {train_loss2:.4f}/{test_loss2:.4f}'
                        )
                
                # Calculate training accuracy for this epoch
                if total_samples > 0:  # Ensure we don't divide by zero
                    train_acc1 = 100 * total_correct1 / total_samples
                    train_acc2 = 100 * total_correct2 / total_samples
                    
                    # Get validation accuracy
                    val_acc1, _ = self.validate_model(self.model1)
                    val_acc2, _ = self.validate_model(self.model2)
                    
                    # Store epoch metrics
                    self.epoch_metrics['model1']['epochs'].append(epoch + 1)
                    self.epoch_metrics['model1']['train_acc'].append(train_acc1)
                    self.epoch_metrics['model1']['val_acc'].append(val_acc1)
                    
                    self.epoch_metrics['model2']['epochs'].append(epoch + 1)
                    self.epoch_metrics['model2']['train_acc'].append(train_acc2)
                    self.epoch_metrics['model2']['val_acc'].append(val_acc2)
                    
                    self.add_log(f"Epoch {epoch + 1} - Model1 Train Acc: {train_acc1:.2f}%, Val Acc: {val_acc1:.2f}%")
                    self.add_log(f"Epoch {epoch + 1} - Model2 Train Acc: {train_acc2:.2f}%, Val Acc: {val_acc2:.2f}%")
                
                self.training_state.current_epoch = epoch + 1
                
        except Exception as e:
            self.add_log(f"Error in training: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
        finally:
            self.training_state.is_training = False
            self.save_final_models()

    def add_log(self, message):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")  # Also print to console

    def get_and_clear_logs(self):
        """Get current logs and clear the log list"""
        logs = self.logs.copy()
        self.logs = []
        return logs

    def evaluate_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / total
        return test_loss, accuracy

    def calculate_epoch_mean_loss(self):
        """Calculate mean losses for the current epoch."""
        train_mean = sum(self.current_epoch_losses['train_loss']) / len(self.current_epoch_losses['train_loss']) if self.current_epoch_losses['train_loss'] else 0
        test_mean = sum(self.current_epoch_losses['test_loss']) / len(self.current_epoch_losses['test_loss']) if self.current_epoch_losses['test_loss'] else 0
        return train_mean, test_mean

    def calculate_accuracy(self, output, target):
        """Calculate accuracy for the current batch."""
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(target)

    def train_epoch(self, epoch, model, optimizer, model_name=""):
        """Training loop for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        processed = 0
        
        # Create a single progress message for the epoch
        self.add_log(f"Starting {model_name} - Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if not self.training_in_progress:
                return 0, 0

            # Get samples and move to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Training step
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            # Log only every 10% of the epoch
            if batch_idx % (len(self.train_loader) // 10) == 0:
                accuracy = 100 * correct / processed
                progress = 100. * batch_idx / len(self.train_loader)
                self.add_log(
                    f"{model_name} - Progress: {progress:.1f}% | "
                    f"Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%"
                )

        # Calculate final metrics
        epoch_loss = running_loss / len(self.train_loader)
        final_accuracy = 100 * correct / len(self.train_loader.dataset)
        
        # Log epoch summary
        self.add_log(
            f"{model_name} - Epoch {epoch + 1} Complete | "
            f"Avg Loss: {epoch_loss:.4f} | "
            f"Final Acc: {final_accuracy:.2f}%"
        )

        return epoch_loss, final_accuracy

    def test(self, model, model_name=""):
        """Evaluation loop"""
        model.train(False)  # Set to evaluation mode
        test_loss = 0
        correct = 0
        
        self.add_log(f"Evaluating {model_name}")
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)

        # Log test results
        self.add_log(
            f"{model_name} - Test Results | "
            f"Avg Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        return test_loss, accuracy

    def save_model(self, path):
        """Save the current model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_acc': self.train_acc[-1] if self.train_acc else 0,
            'test_acc': self.test_acc[-1] if self.test_acc else 0,
            'epoch': len(self.training_accuracies)
        }, path)

    def load_model(self, path):
        """Load a saved model state"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
        return None

    def reset(self):
        """Reset all training state"""
        try:
            # Reset models
            self.model1 = Net().to(self.device)
            self.model2 = Net().to(self.device)
            self.optimizer1 = None
            self.optimizer2 = None
            
            # Reset all metrics and state
            self.training_in_progress = False
            self.training_losses1 = []
            self.training_losses2 = []
            self.training_accuracies1 = []
            self.training_accuracies2 = []
            self.logs = []
            
            self.add_log("Training state has been completely reset")
            return True
        except Exception as e:
            self.add_log(f"Error during reset: {str(e)}")
            raise e

    def get_sample_predictions(self):
        """Get exactly 5 wrong and 5 right predictions for each model"""
        try:
            if not hasattr(self, 'model1') or self.model1 is None or not hasattr(self, 'model2') or self.model2 is None:
                print("Models not initialized")
                return None

            samples = {
                'model1': {'wrong_samples': [], 'correct_samples': []},
                'model2': {'wrong_samples': [], 'correct_samples': []}
            }

            # Keep fetching batches until we have enough samples
            while (len(samples['model1']['wrong_samples']) < 5 or 
                   len(samples['model1']['correct_samples']) < 5 or
                   len(samples['model2']['wrong_samples']) < 5 or
                   len(samples['model2']['correct_samples']) < 5):
                
                # Get a batch of test data
                data_iter = iter(self.test_loader)
                images, targets = next(data_iter)
                images, targets = images.to(self.device), targets.to(self.device)

                # Get predictions from both models
                self.model1.eval()
                self.model2.eval()
                with torch.no_grad():
                    outputs1 = self.model1(images)
                    outputs2 = self.model2(images)
                    predictions1 = outputs1.argmax(dim=1)
                    predictions2 = outputs2.argmax(dim=1)

                # Process predictions for both models
                for i in range(len(targets)):
                    # Convert tensor to PIL Image
                    img = images[i].cpu()
                    img = (img - img.min()) / (img.max() - img.min())
                    img = transforms.ToPILImage()(img)
                    
                    # Convert PIL image to base64 string
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Create sample data
                    sample = {
                        'image': img_str,
                        'actual': targets[i].item()
                    }

                    # Process Model 1 predictions
                    if len(samples['model1']['wrong_samples']) < 5 or len(samples['model1']['correct_samples']) < 5:
                        sample1 = sample.copy()
                        sample1['predicted'] = predictions1[i].item()
                        if predictions1[i] == targets[i]:
                            if len(samples['model1']['correct_samples']) < 5:
                                samples['model1']['correct_samples'].append(sample1)
                        else:
                            if len(samples['model1']['wrong_samples']) < 5:
                                samples['model1']['wrong_samples'].append(sample1)

                    # Process Model 2 predictions
                    if len(samples['model2']['wrong_samples']) < 5 or len(samples['model2']['correct_samples']) < 5:
                        sample2 = sample.copy()
                        sample2['predicted'] = predictions2[i].item()
                        if predictions2[i] == targets[i]:
                            if len(samples['model2']['correct_samples']) < 5:
                                samples['model2']['correct_samples'].append(sample2)
                        else:
                            if len(samples['model2']['wrong_samples']) < 5:
                                samples['model2']['wrong_samples'].append(sample2)

                # Break if we have enough samples
                if (len(samples['model1']['wrong_samples']) == 5 and 
                    len(samples['model1']['correct_samples']) == 5 and
                    len(samples['model2']['wrong_samples']) == 5 and
                    len(samples['model2']['correct_samples']) == 5):
                    break

            return samples

        except Exception as e:
            print(f"Error in get_sample_predictions: {str(e)}")
            return None
    
    def store_metrics(self, epoch, train_loss1, test_loss1, train_accuracy1, test_accuracy1,
                     train_loss2, test_loss2, train_accuracy2, test_accuracy2):
        # Store metrics for Model 1
        self.training_losses1.append({
            'epoch': epoch,
            'train_loss': train_loss1,
            'test_loss': test_loss1
        })
        self.training_accuracies1.append({
            'epoch': epoch,
            'train_accuracy': train_accuracy1,
            'test_accuracy': test_accuracy1
        })

        # Store metrics for Model 2
        self.training_losses2.append({
            'epoch': epoch,
            'train_loss': train_loss2,
            'test_loss': test_loss2
        })
        self.training_accuracies2.append({
            'epoch': epoch,
            'train_accuracy': train_accuracy2,
            'test_accuracy': test_accuracy2
        })

    def get_weights_histogram(self):
        # Return weights histogram data
        return [param.data.cpu().numpy().flatten() for param in self.model.parameters()]

    def get_biases_histogram(self):
        # Return biases histogram data
        return [param.data.cpu().numpy().flatten() for param in self.model.parameters() if param.dim() == 1]

    def get_gradients_histogram(self):
        # Return gradients histogram data
        return [param.grad.data.cpu().numpy().flatten() for param in self.model.parameters() if param.grad is not None]

    def get_embeddings(self):
        # Return embeddings (dummy implementation)
        return {"embeddings": "dummy_embeddings"}

    def get_performance_stats(self):
        # Return performance stats (dummy implementation)
        return {"gpu_usage": "dummy_usage", "memory_usage": "dummy_memory"}

    def get_hparams(self):
        # Return hyperparameters (dummy implementation)
        return {"learning_rate": 0.01, "batch_size": 128}

    def validate_model(self, model):
        """Validate the model and return accuracy and loss"""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += F.nll_loss(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy, val_loss / len(self.test_loader)

    def save_model_if_better(self, model, accuracy, model_num):
        """Save model if it has better accuracy"""
        if model_num == 1:
            if accuracy > self.best_accuracy1:
                self.best_accuracy1 = accuracy
                self.save_model(model, f'best_model1.pth')
                self.add_log(f'Saved new best model1 with accuracy: {accuracy:.2f}%')
        else:
            if accuracy > self.best_accuracy2:
                self.best_accuracy2 = accuracy
                self.save_model(model, f'best_model2.pth')
                self.add_log(f'Saved new best model2 with accuracy: {accuracy:.2f}%')

    def save_model(self, model, filename):
        """Save model with additional information"""
        save_path = self.models_dir / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': self.best_accuracy1 if 'model1' in filename else self.best_accuracy2,
            'epoch': self.training_state.current_epoch,
            'optimizer_state_dict': self.optimizer1.state_dict() if 'model1' in filename else self.optimizer2.state_dict(),
            'model_config': {
                'conv1_channels': model.conv1.out_channels,
                'conv2_channels': model.conv2.out_channels,
                'conv3_channels': model.conv3.out_channels
            }
        }, save_path)

    def save_final_models(self):
        """Save final versions of both models"""
        self.save_model(self.model1, 'final_model1.pth')
        self.save_model(self.model2, 'final_model2.pth')
        self.add_log('Saved final models')

    def load_model(self, filename):
        """Load a saved model"""
        try:
            checkpoint = torch.load(self.models_dir / filename)
            model = Net(
                conv1_channels=checkpoint['model_config']['conv1_channels'],
                conv2_channels=checkpoint['model_config']['conv2_channels'],
                conv3_channels=checkpoint['model_config']['conv3_channels']
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint['accuracy']
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, 0

    def compare_saved_models(self):
        """Compare saved models and return the best one"""
        models = {
            'best_model1.pth': None,
            'best_model2.pth': None,
            'final_model1.pth': None,
            'final_model2.pth': None
        }
        
        best_accuracy = 0
        best_model_name = None
        
        for model_file in models.keys():
            model, accuracy = self.load_model(model_file)
            if model is not None:
                models[model_file] = (model, accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_file
        
        return {
            'models': models,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy
        }

    def get_metrics(self):
        """Get current training metrics"""
        return {
            'epoch_metrics': self.epoch_metrics,
            'batch_metrics': self.batch_metrics
        }

    def train_batch(self, model, optimizer, data, target):
        """Train one batch and return loss"""
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = self.criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_test_batch_loss(self, model):
        """Get test loss for current state of model"""
        model.eval()
        test_loss = 0
        with torch.no_grad():
            # Get one batch from test loader
            try:
                data, target = next(iter(self.test_loader))
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss = self.criterion(output, target).item()
            except StopIteration:
                # Reset test loader if we've gone through all batches
                self.setup_data_loaders()
                data, target = next(iter(self.test_loader))
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss = self.criterion(output, target).item()
        return test_loss

    def reset_model(self):
        """Reset both models to initial state"""
        try:
            # Reinitialize models with current configurations
            self.model1 = Net(
                conv1_channels=self.model1.conv1_channels,
                conv2_channels=self.model1.conv2_channels,
                conv3_channels=self.model1.conv3_channels
            ).to(self.device)
            
            self.model2 = Net(
                conv1_channels=self.model2.conv1_channels,
                conv2_channels=self.model2.conv2_channels,
                conv3_channels=self.model2.conv3_channels
            ).to(self.device)
            
            # Reset optimizers
            self.optimizer1 = None
            self.optimizer2 = None
            
            # Reset batch counter
            self.current_batch = 0
            
            print("Models reset successfully")
        except Exception as e:
            print(f"Error in reset_model: {str(e)}")
            raise

    def reset_metrics(self):
        """Reset all training metrics"""
        try:
            # Reset all metrics storage
            self.training_losses1 = []
            self.training_losses2 = []
            self.training_accuracies1 = []
            self.training_accuracies2 = []
            
            # Reset epoch metrics
            self.epoch_metrics = {
                'model1': {
                    'train_acc': [],
                    'val_acc': [],
                    'epochs': []
                },
                'model2': {
                    'train_acc': [],
                    'val_acc': [],
                    'epochs': []
                }
            }
            
            # Reset batch metrics
            self.batch_metrics = {
                'model1': {
                    'train_loss': [],
                    'test_loss': [],
                    'batch_numbers': []
                },
                'model2': {
                    'train_loss': [],
                    'test_loss': [],
                    'batch_numbers': []
                }
            }
            
            # Reset training state
            self.training_state.current_epoch = 0
            self.best_accuracy1 = 0
            self.best_accuracy2 = 0
            
            print("Metrics reset successfully")
        except Exception as e:
            print(f"Error in reset_metrics: {str(e)}")
            raise
