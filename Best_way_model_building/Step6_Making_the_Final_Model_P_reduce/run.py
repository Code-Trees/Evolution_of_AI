import warnings
warnings.filterwarnings('ignore')

import torch
from model.model import Net
from utils.optimizer import get_optimizer,run_lrfinder
from utils.model_fit import training,testing
from utils.data_loader import MNISTDataLoader
from torch import nn
import json
import plotext as plt
import numpy as np
import random
import os
import sys
from rf_calc import receptive_field
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
import torchvision
from itertools import islice
import datetime
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# torch.torch.set_default_dtype(torch.float64)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using torch.backends.cudnn, set the following for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(device):
    model = Net().to(device)
    # model = model.float()
    return model


def run_model(model,device,batch_size,epochs,optimizer,scheduler,use_scheduler,best_model,version="v1",notes=""):
    # Create unique run name with version and timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"experiment_{version}_{timestamp}"
    
    # Comment out TensorBoard writer initialization and logging
    # writer = SummaryWriter(f'runs/{run_name}')
    # writer.add_text('Model/Scheduler', str(scheduler) if use_scheduler else "None")
    # writer.add_text('Training/BatchSize', str(batch_size))
    # writer.add_text('Training/Epochs', str(epochs))
    
    current_lr = optimizer.param_groups[0]['lr']
    
    # Comment out model architecture logging
    # dummy_input = torch.zeros((1, 1, 28, 28)).to(device)
    # writer.add_graph(model, dummy_input)
    
    # Comment out transform logging function
    # def log_transforms():
    #     images, labels = next(iter(train_loader))
    #     img_grid = torchvision.utils.make_grid(images[:25])
    #     writer.add_image('mnist_images', img_grid)
    #     transform_params = str(data_loader.train_transforms)
    #     writer.add_text('transforms/train', transform_params)
    
    # log_transforms()
    
    train_losses = []
    train_accuracy = []
    test_losses =[]
    test_accuracy = []
    # print(scheduler)
    # print(optimizer)
    print(summary(model, (1,28, 28 )))
    # _ = receptive_field(model,28)

    # Comment out experiment metadata logging
    # writer.add_text('Experiment/Version', version)
    # writer.add_text('Experiment/DateTime', timestamp)
    # writer.add_text('Experiment/Notes', notes)
    
    for EPOCHS in range(0,epochs):
        train_loss, train_acc = training(model,device,train_loader,optimizer,EPOCHS)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        test_loss,test_acc = testing(model,device,test_loader,EPOCHS)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
        
        # Comment out tensorboard logging in training loop
        # writer.add_scalars('Loss', {
        #     'train': train_loss,
        #     'test': test_loss
        # }, EPOCHS)
        
        # writer.add_scalars('Accuracy', {
        #     'train': train_acc,
        #     'test': test_acc
        # }, EPOCHS)
        
        # writer.add_scalar('Learning_Rate', current_lr, EPOCHS)
        
        # Comment out model weights histogram logging
        # if EPOCHS % 5 == 0:
        #     for name, param in model.named_parameters():
        #         writer.add_histogram(f'Parameters/{name}', param.data, EPOCHS)
        #         if param.grad is not None:
        #             writer.add_histogram(f'Gradients/{name}', param.grad, EPOCHS)
        
        if (scheduler_type == 'reducelronplateau') & (use_scheduler ==True):
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
        elif (scheduler_type == 'steplr') & (use_scheduler ==True):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        try:
            if len(test_accuracy) > 1:
                if (EPOCHS >= 3 and 
                    max(test_accuracy[:-1]) < test_accuracy[-1] and 
                    max(test_accuracy) >= best_model):
                    
                    checkpoint = {
                        'epoch': EPOCHS + 1,
                        'valid_loss_min': test_losses[-1],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    
                    file_name = f"./model_folder/modelbest_{test_accuracy[-1]:.4f}_epoch_{EPOCHS}.pt"
                    torch.save(checkpoint, file_name)
                    print(f"Target Achieved: {max(test_accuracy) * 100:.2f}% Test Accuracy!!")
                else:
                    print("Conditions not met for saving the model.")
            else:
                print("Insufficient test accuracy data.")
        except Exception as e:
            print(f"Model saving failed: {e}")

        print(f"LR: {current_lr}\n")
    # Comment out final metrics logging
    # writer.add_hparams(
    #     {
    #         'batch_size': batch_size,
    #         'epochs': epochs,
    #         'optimizer': type(optimizer).__name__,
    #         'scheduler': str(scheduler_type) if use_scheduler else 'None',
    #         'learning_rate': lrs[0]
    #     },
    #     {
    #         'hparam/best_train_acc': max(train_accuracy),
    #         'hparam/best_test_acc': max(test_accuracy),
    #         'hparam/final_train_loss': train_losses[-1],
    #         'hparam/final_test_loss': test_losses[-1]
    #     }
    # )
    
    # writer.close()
    return model,train_losses, train_accuracy,test_losses,test_accuracy


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def get_loss_function(loss_type):
    if loss_type is None:
        return nn.NLLLoss()   
    loss_types = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'nll': nn.NLLLoss()
    }
    return loss_types.get(loss_type.lower(), nn.CrossEntropyLoss())


if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', type=str, default='v1', help='Version of the experiment')
    args = parser.parse_args()
    
    config = load_config()
    
    # Set seed from config
    set_seed(config['seed'])
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get loss function and scheduler settings from config
    loss_fn = get_loss_function(config['training'].get('loss_type'))
    use_scheduler = config['training'].get('use_scheduler', False)
    scheduler_type = config['training'].get('scheduler_type', 'steplr')
    runlr_finer = config['training'].get('runlr_finer', False)
    use_scheduler = bool(use_scheduler)
    runlr_finer = bool(runlr_finer)

    best_model = config['best_model']
    version = args.v  # Use version from command line argument
    notes = config['training'].get('notes', '')

    # Set seed from config
    _ = torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(config['seed'])
    
    # Get batch size based on device
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    model = get_model(device)
    data_loader = MNISTDataLoader(batch_size=batch_size)
    train_loader, test_loader = data_loader.get_data_loaders()

    if runlr_finer:
        lrs,_ = run_lrfinder(
            model, 
            device, 
            train_loader, 
            test_loader, 
            start_lr=config['training']['start_lr'],
            end_lr=config['training']['end_lr'],
                lr_iter=config['training'].get('lr_iter', 1000)
            )
        print(lrs)
    else:
        lrs = [0.015]

    optimizer,scheduler = get_optimizer(model,scheduler = use_scheduler,\
                              scheduler_type = scheduler_type,lr = lrs[0])

    model,train_losses, train_accuracy,test_losses,test_accuracy= run_model(model,device,batch_size,epochs,optimizer,scheduler,use_scheduler,best_model,version,notes)


    print("Max Train Accuracy: ",max(train_accuracy))
    print("Max Test Accuracy: ",max(test_accuracy))

    # # Set a canvas size suitable for your terminal
    # plt.plotsize(5,5)

    # # Plot for accuracy
    # plt.clf()  # Clear the canvas before starting a new plot
    # plt.plot(train_accuracy, label="Train Accuracy")
    # plt.plot(test_accuracy, label="Test Accuracy")
    # plt.title("Model Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.show()
    
    # plt.plotsize(5,5)
    # # Plot for loss
    # plt.clf()  # Clear the canvas again for the next plot
    # plt.plot(train_losses, label="Train Loss")
    # plt.plot(test_losses, label="Test Loss")
    # plt.title("Model Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

