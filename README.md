# ImCaption
Image to Caption Task

# Code Organization
<pre>

main.py - Main script to train and test model
model.py - Architectures
utils - Utilities functions
dataloader.py - dataloading and other related utils
vocabulary.py - vocabulary preprocessing

</pre>

# Running the code
In order to run the code, there are several flags to set understand. The 
code is structured for users, who would like to train/test a model and/or to
 run several experiments and make comparisons. 
 
 ## Train/Test Model
 To train or test the model at its most basic configurations, simply run 
```python main.py```, while setting ```mode``` to ```'train'```,  ```'eval'```
or ```'deploy'```. Below are 
the 
default 
configurations, which can 
be adjusted at your own desire. 

<pre>
# Module level constants
MODE = 'train'
CNN_ARCH = 'resnet50'
MODEL_DIR = './models/test-model'
RESUME = None

# Set values for the training variables
batch_size = 32  # batch size
vocab_threshold = 5  # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 256  # dimensionality of image and word embeddings
hidden_size = 512  # number of features in hidden state of the RNN decoder
num_epochs = 10  # number of training epochs
lr = 0.001 # learning rate
estop_threshold = 3 # early stop threshold
attention = True
alpha_c = 1.
</pre>