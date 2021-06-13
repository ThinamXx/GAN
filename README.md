### **DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS**

**Generative Adversarial Networks**
- Generative Adversarial Networks consist of two deep networks Generator and Discriminator. The Generator generates the image as much closer to the true image as possible to fool Discriminator by maximizing the cross entropy loss. The Discriminator tries to distinguish the generated images from the true images by minimizing the cross entropy loss.

**Libraries and Dependencies**
- I have downloaded all the libraries and dependencies required for the project in one particular cell.

```javascript
from d2l import torch as d2l
import warnings
import torch   
import torchvision  
from torch import nn
```

**The Pokemon Dataset**
- The dataset is a collection of Pokemon sprites obtained from PokemonDB. I will download, extract and load the dataset. I will resize each image into 64X64 and normalize the data with 0.5 mean and 0.5 standard deviation. I have presented the implementation of Pokemon Dataset using PyTorch here in the snapshots.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20188b.PNG)

**The Generator**
- The Generator needs to map the noise variable to a RGB image. I will use transposed convolutional layer to enlarge the input image. The basic block of Generator contains a transposed convolution layer followed by the batch normalization and RELU activation function. The Generator consists of four blocks that increase height and width of input from 1 to 32. The transposed convolution layer is used to generate the output. The tanh activation function is used to project output values into the range of -1 and 1. I have presented the implementation of The Generator Block and Pokemon Dataset using PyTorch here in the snapshots.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20188a.PNG)

**The Discriminator**
- The Discriminator is a convolution layer followed by a batch normalization layer and Leaky RELU activation function. Leaky RELU is a nonlinear function that gives a non zero output for a negative input. It aims to fix the RELU problem that a neuron might always output a negative value and therefore cannot make any progress since the gradient of RELU is 0. I have presented the implementation of The Discriminator Block and The Generator Block using PyTorch here in the snapshots.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20189.PNG)

**Training**
- I will be using the same learning rate for both generator and discriminator since the networks are similar to each other. I will change β1 in Adam from 0.9 to 0.5. It decreases the smoothness of the momentum which is the exponentially weighted moving average of past gradients to take care of the rapid changing gradients because the generator and the discriminator fight with each other. The random generated noise Z is a 4D tensor. I have presented the implementation of Training Generator and Discriminator Networks using PyTorch here in the snapshots. 

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20190a.PNG)
![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20190b.PNG)
