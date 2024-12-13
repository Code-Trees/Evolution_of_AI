# Target: Achieve 99.4 in Less than 20 epochs and 10k parameters

![Build Status](https://github.com/Code-Trees/mnist_ops/actions/workflows/python-app.yml/badge.svg)

## Model Architecture

This deep learning model is a Convolution Neural Network (CNN) designed for image classification tasks. The architecture is inspired by many people with additional modifications for improved performance.

### Key Features:
- Automatic data scaling
- Batch normalization layers for stable training
- Dropout layers for regularization
- Softmax activation for multi-class classification
- Less parameter-based model, best performance
- LR-Finder / Scheduler for faster training
- Modularized code for easy maintenance
- Test cases for GitHub action deployment
- CPU and GPU-based training

### Architecture Details:

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
           Dropout-3           [-1, 10, 26, 26]               0
              ReLU-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             900
       BatchNorm2d-6           [-1, 10, 24, 24]              20
           Dropout-7           [-1, 10, 24, 24]               0
              ReLU-8           [-1, 10, 24, 24]               0
            Conv2d-9           [-1, 10, 22, 22]             900
      BatchNorm2d-10           [-1, 10, 22, 22]              20
          Dropout-11           [-1, 10, 22, 22]               0
             ReLU-12           [-1, 10, 22, 22]               0
           Conv2d-13           [-1, 10, 20, 20]             900
      BatchNorm2d-14           [-1, 10, 20, 20]              20
          Dropout-15           [-1, 10, 20, 20]               0
             ReLU-16           [-1, 10, 20, 20]               0
        MaxPool2d-17           [-1, 10, 10, 10]               0
           Conv2d-18             [-1, 10, 8, 8]             900
      BatchNorm2d-19             [-1, 10, 8, 8]              20
          Dropout-20             [-1, 10, 8, 8]               0
             ReLU-21             [-1, 10, 8, 8]               0
           Conv2d-22             [-1, 16, 6, 6]           1,440
      BatchNorm2d-23             [-1, 16, 6, 6]              32
          Dropout-24             [-1, 16, 6, 6]               0
             ReLU-25             [-1, 16, 6, 6]               0
           Conv2d-26             [-1, 16, 4, 4]           2,304
      BatchNorm2d-27             [-1, 16, 4, 4]              32
          Dropout-28             [-1, 16, 4, 4]               0
             ReLU-29             [-1, 16, 4, 4]               0
        AvgPool2d-30             [-1, 16, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]             160
================================================================
Total params: 7,758
Trainable params: 7,758
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.03
Estimated Total Size (MB): 0.74
----------------------------------------------------------------
```

### Model Parameters
- Total Parameters: 7.7K
- Trainable Parameters: 7.7K
- Input Shape: `(1, 28, 28)`
- Output Classes: 10

## Data Augmentation Pipeline

### Image Augmentation Techniques

![Screenshot from 2024-11-29 20-50-15](readme_images/train_transform.png)

After Looking at data I have comeup with this Augmentation . Also The sequence of augmentation matters because 

1. **`transforms.Resize((28, 28))`**:
   - Ensures all images are resized to a uniform size (28x28), which is often required for neural networks to maintain consistent input dimensions.
   - Interpolation mode like `NEAREST` helps retain sharp edges, suitable for datasets like MNIST.
2. **(Optional) `transforms.ColorJitter`**:
   - Introduces variations in brightness, contrast, saturation, and hue to make the model robust to different lighting conditions (currently commented out in your code).
3. **`transforms.RandomPerspective`**:
   - Applies a random perspective distortion to simulate variations in viewpoint or perspective.
   - Adds diversity to training data and improves generalization, especially for datasets with handwritten or irregular shapes.
4. **`transforms.RandomRotation`**:
   - Rotates the image randomly within a specified range (±15° in this case).
   - Helps the model handle rotated versions of objects, useful for tasks where orientation might vary.
   - The `fill` parameter ensures the new background introduced during rotation is filled with a specified value (like `self.mean`).
5. **`transforms.ToTensor()`**:
   - Converts the image into a PyTorch tensor and scales pixel values to [0,1][0,1].
6. **`transforms.Normalize((self.mean,), (self.std,))`**:
   - Normalizes the image tensor to have a mean of `0` and a standard deviation of `1`, using the dataset's `mean` and `std`.
   - Ensures consistent input range and accelerates convergence during training.

## Model Running Pipeline

### LR Finders

for 100 Iter:

```python
 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 3739/4500 [00:32<00:06, 117.91it/s]Stopping early, the loss has diverged
 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 3741/4500 [00:32<00:06, 114.63it/s]
Learning rate search finished. See the graph with {finder_name}.plot()
LR suggestion: steepest gradient
Suggested LR: 1.01E-01
Loss: 0.07616443393017538 LR: 0.10080224034423935
[0.10080224034423935]
```

![Alt text](readme_images/download.png)


The model will automatic pick the Best learning rate . For Small models like MNIST it is not so important. But for bigger model it is going to save real money by reducing your training time 

### Training Epochs

```python
None
=======================================Reciptive Field Calculator========================================
|    | Kernel_size   | Padding   |   Stride | Input_Img_size   | Output_Img_size   | Receptive_field   |
|---:|:--------------|:----------|---------:|:-----------------|:------------------|:------------------|
|  0 | 3*3           | NO        |        1 | 28*28            | 26*26             | 3*3               |
|  1 | 3*3           | NO        |        1 | 26*26            | 24*24             | 5*5               |
|  2 | 3*3           | NO        |        1 | 24*24            | 22*22             | 7*7               |
|  3 | 2*2           | NO        |        2 | 22*22            | 11*11             | 8*8               |
|  4 | 1*1           | NO        |        1 | 11*11            | 11*11             | 8*8               |
|  5 | 3*3           | NO        |        1 | 11*11            | 9*9               | 12*12             |
|  6 | 3*3           | NO        |        1 | 9*9              | 7*7               | 16*16             |
|  7 | 3*3           | NO        |        1 | 7*7              | 5*5               | 20*20             |
|  8 | 5*5           | NO        |        1 | 5*5              | 1*1               | 28*28             |
=========================================================================================================
Train ==> Epochs: 0 Batch:  468 loss: 0.09212353080511093 Accuracy: 92.77% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 111.87it/s]
Test ==> Epochs: 0 Batch:  78 loss: 0.05814522932767868 Accuracy: 98.22% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 182.62it/s]
Insufficient test accuracy data.
LR: 0.10080224034423935

Train ==> Epochs: 1 Batch:  468 loss: 0.11005734652280807 Accuracy: 96.95% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 110.61it/s]
Test ==> Epochs: 1 Batch:  78 loss: 0.04768860439956188 Accuracy: 98.58% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 196.15it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 2 Batch:  468 loss: 0.07364116609096527 Accuracy: 97.26% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 109.20it/s]
Test ==> Epochs: 2 Batch:  78 loss: 0.03464922721385956 Accuracy: 98.89% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 187.17it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 3 Batch:  468 loss: 0.007149364799261093 Accuracy: 97.59% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 108.47it/s]
Test ==> Epochs: 3 Batch:  78 loss: 0.030723155471868813 Accuracy: 98.97% : 100%|████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 185.69it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 4 Batch:  468 loss: 0.08439592272043228 Accuracy: 97.78% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 109.07it/s]
Test ==> Epochs: 4 Batch:  78 loss: 0.03019285760708153 Accuracy: 98.95% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 171.26it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 5 Batch:  468 loss: 0.046797674149274826 Accuracy: 97.84% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 110.89it/s]
Test ==> Epochs: 5 Batch:  78 loss: 0.024491356252133847 Accuracy: 99.18% : 100%|████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 188.11it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 6 Batch:  468 loss: 0.015085098333656788 Accuracy: 97.96% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 110.69it/s]
Test ==> Epochs: 6 Batch:  78 loss: 0.0314102898158133 Accuracy: 98.96% : 100%|██████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 179.05it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 7 Batch:  468 loss: 0.2118207961320877 Accuracy: 98.06% : 100%|██████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 109.24it/s]
Test ==> Epochs: 7 Batch:  78 loss: 0.025353129401803016 Accuracy: 99.20% : 100%|████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 174.53it/s]
Conditions not met for saving the model.
LR: 0.10080224034423935

Train ==> Epochs: 8 Batch:  468 loss: 0.02431732416152954 Accuracy: 98.02% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 109.18it/s]
Test ==> Epochs: 8 Batch:  78 loss: 0.025130902411043644 Accuracy: 99.25% : 100%|████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 195.03it/s]
Conditions not met for saving the model.
LR: 0.010080224034423935

Train ==> Epochs: 9 Batch:  468 loss: 0.01332076545804739 Accuracy: 98.44% : 100%|█████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 108.46it/s]
Test ==> Epochs: 9 Batch:  78 loss: 0.02061711220755242 Accuracy: 99.36% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 185.94it/s]
Target Achieved: 99.36% Test Accuracy!!
LR: 0.010080224034423935

Train ==> Epochs: 10 Batch:  468 loss: 0.015133660286664963 Accuracy: 98.52% : 100%|███████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 110.33it/s]
Test ==> Epochs: 10 Batch:  78 loss: 0.02010468928143382 Accuracy: 99.39% : 100%|████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 179.79it/s]
Target Achieved: 99.39% Test Accuracy!!
LR: 0.010080224034423935

Train ==> Epochs: 11 Batch:  468 loss: 0.05695578455924988 Accuracy: 98.66% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 107.37it/s]
Test ==> Epochs: 11 Batch:  78 loss: 0.0187287402221933 Accuracy: 99.48% : 100%|█████████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 176.32it/s]
Target Achieved: 99.48% Test Accuracy!!
LR: 0.010080224034423935

Train ==> Epochs: 12 Batch:  468 loss: 0.07540423423051834 Accuracy: 98.56% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 104.31it/s]
Test ==> Epochs: 12 Batch:  78 loss: 0.018561058009415866 Accuracy: 99.43% : 100%|███████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 180.08it/s]
Conditions not met for saving the model.
LR: 0.010080224034423935

Train ==> Epochs: 13 Batch:  468 loss: 0.04020127281546593 Accuracy: 98.61% : 100%|████████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 106.60it/s]
Test ==> Epochs: 13 Batch:  78 loss: 0.019587678169459104 Accuracy: 99.41% : 100%|███████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 180.39it/s]
Conditions not met for saving the model.
LR: 0.010080224034423935

Train ==> Epochs: 14 Batch:  468 loss: 0.022925393655896187 Accuracy: 98.56% : 100%|███████████████████████████████████████████████████████████████████████| 469/469 [00:04<00:00, 107.54it/s]
Test ==> Epochs: 14 Batch:  78 loss: 0.018888854747172446 Accuracy: 99.38% : 100%|███████████████████████████████████████████████████████████████████████████| 79/79 [00:00<00:00, 176.43it/s]
Conditions not met for saving the model.
LR: 0.0010080224034423936

Max Train Accuracy:  0.9865666666666667
Max Test Accuracy:  0.9948

```

- Learning rate annealing helped prevent over fitting.

![Alt text](readme_images/Train_test.png)

## Why Model Not able to predict 100% of data in test set .

After looking at data , I have realized that the Training set and test set have some data Which is difficult to predict by human eye . 

That is where the model is learning garbage. But This doest the work  and Fulfill out target,

#### Train Model Wrong Predicted Images with 40 lowest confidence

![Train_model_image](readme_images/Train_model_image.png)

#### Test model Wrong Predicted Images with  40 lowest confidence 

![Test_model_image](readme_images/Test_model_image.png)

### Deployment

**Test cases**

Test case help to validate that I am pushing Correct information to  repo. Let's test it locally:

```python
============================= test session starts ==============================
platform linux -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0 -- /bin/python
cachedir: .pytest_cache
rootdir: Mnist_ops
plugins: anyio-4.6.2.post1
collected 11 items                                                             

tests/test_model.py::test_model_param_count PASSED                       [  9%]
tests/test_model.py::test_model_output_shape PASSED                      [ 18%]
tests/test_model.py::test_cuda_available PASSED                          [ 27%]
tests/test_model.py::test_batch_size PASSED                              [ 36%]
tests/test_model.py::test_calculate_stats PASSED                         [ 45%]
tests/test_model.py::test_transformations PASSED                         [ 54%]
tests/test_model.py::test_dataloader_args PASSED                         [ 63%]
tests/test_model.py::test_data_loaders PASSED                            [ 72%]
tests/test_model.py::test_data_augmentation PASSED                       [ 81%]
tests/test_model.py::test_training PASSED                                [ 90%]
tests/test_model.py::test_training_with_scheduler PASSED                 [100%]

============================= 11 passed in 36.37s ==============================
```



## Push to git hub with Git action configured

![GithubCommit](readme_images/Gitlog1.png)







![GithubAction](readme_images/Git_log.png)

**Git Action logs**

**LR finder**

 94%|█████████▍| 945/1000 [01:04<00:03, 14.64it/s]

Stopping early, the loss has diverged

Learning rate search finished. See the graph with {finder_name}.plot()

LR suggestion: steepest gradient

Suggested LR: 3.12E-02

Loss: 0.31175027199026456 LR :0.031152542235554845



**Training/Testing Loop:**

Train ==> Epochs: 0 Batch:  937 loss: 0.03763921186327934 Accuracy: 90.91% : 100%|██████████| 938/938 [01:05<00:00, 15.51it/s]

Train ==> Epochs: 0 Batch:  937 loss: 0.03763921186327934 Accuracy: 90.91% : 100%|██████████| 938/938 [01:05<00:00, 14.35it/s]



Test ==> Epochs: 0 Batch:  156 loss: 0.0016227789369411766 Accuracy: 96.75% : 100%|██████████| 157/157 [00:08<00:00, 19.35it/s]

Insufficient test accuracy data.

LR: 0.031152542235554845

Max Train Accuracy:  0.9090833333333334

Max Test Accuracy:  0.9675





## Requirements

- torch>=2.4.1 --index-url https://download.pytorch.org/whl/cpu
- torchvision>=0.19.1 --index-url https://download.pytorch.org/whl/cpu
- Albumentations 1.1.0
- NumPy 1.21+
- OpenCV 4.5+
- CUDA 11.3+ (for GPU support ONLY)

