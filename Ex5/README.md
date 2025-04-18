<h1>Object 5</h1>
<h4> WAP  to  train  and  evaluate  a  convolutional  neural  network  using  Keras  Library  to 
classify  MNIST  fashion  dataset.  Demonstrate  the  effect  of  filter  size,  regularization, 
batch size and optimization algorithm on model performance. </h4>
<hr>


<h3>Description of the model:-</h3>
<b>1. Data Set : </b><p>The dataset used in the training of the model is "MNIST Fashion" which contains around "70000" images out of which "60000" are used for Training data and rest for the "Validation Data".

<b>2. Model Architecture : </b><p>Below is the table illustration the architecture of the model including different types of layers, their types, shapes and their description. </p>

<table border="1" cellspacing="0" cellpadding="10">
    <tr>
        <th>Layer</th>
        <th>Type</th>
        <th>Output Shape</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><b>Input Layer</b></td>
        <td>2D Image (28×28×1)</td>
        <td>(28,28,1)</td>
        <td>Grayscale image input.</td>
    </tr>
    <tr>
        <td><b>Conv2D-1</b></td>
        <td>Convolution (32 filters)</td>
        <td>(26,26,32)</td>
        <td>Applies 32 filters of size (filter_size x filter_size) (default: 3x3) with ReLU activation.</td>
    </tr>
    <tr>
        <td><b>MaxPooling-1</b></td>
        <td>Max Pooling (2×2)</td>
        <td>(13,13,32)</td>
        <td>Reduces spatial dimensions by 2x2 pooling.</td>
    </tr>
    <tr>
        <td><b>Conv2D-2</b></td>
        <td>Convolution (64 filters)</td>
        <td>(11,11,64)</td>
        <td>Applies 64 filters of size (filter_size x filter_size) (default: 3x3) with ReLU activation.</td>
    </tr>
    <tr>
        <td><b>MaxPooling-2</b></td>
        <td>Max Pooling (2×2)</td>
        <td>(5,5,64)</td>
        <td>Again reduces spatial dimensions by 2x2 pooling.</td>
    </tr>
    <tr>
        <td><b>Flatten</b></td>
        <td>Flatten Layer</td>
        <td>(1600)</td>
        <td>Converts the 2D feature maps into a 1D vector.</td>
    </tr>
    <tr>
        <td><b>Dense-1</b></td>
        <td>Fully Connected (128)</td>
        <td>(128)</td>
        <td>Fully connected layer with 128 neurons and ReLU activation.</td>
    </tr>
    <tr>
        <td><b>Dense-2 (Output)</b></td>
        <td>Fully Connected (10)</td>
        <td>(10)</td>
        <td>Outputs 10 classes (one for each clothing type) with softmax activation.</td>
    </tr>
</table>

<b>3. Hyperparameters :</b><p>Although the model is trained and evaluated against the varying values of various hyperparameter, below shows the default hyperparameter configuration</p>
<table border="1" cellspacing="0" cellpadding="10">
    <tr>
        <th>Hyperparameter</th>
        <th>Default Value</th>
    </tr>
    <tr>
        <td><b>Filter Size</b></td>
        <td>3x3</td>
    </tr>
    <tr>
        <td><b>Regularization</b></td>
        <td>None</td>
    </tr>
    <tr>
        <td><b>Optimizer</b></td>
        <td>Adam</td>
    </tr>
    <tr>
        <td><b>Batch Size</b></td>
        <td>32</td>
    </tr>
</table>

<b>4. Experimental Variations : </b><p>
The model is tested with different hyperparameters : 
<ul>
<li>Batch Size : 3x3 vs 5x5</li>
<li>Regularization : None vs L2</li>
<li>Batch Size : 32 vs 64</li>
<li>Optimizers: Adam vs SGD</li>
</ul>
<hr>
<h3>Description of the code:-</h3> 
<ol>
<li><b>Import Required Libraries : </b><ul>
<li>TensorFlow/Keras: Used to build and train the CNN model.</li>
<li>Matplotlib & NumPy: Used for visualization and numerical operations.</li>
</ul></li><br>

<li><b>Enable GPU Acceleration (if available): </b>Checks if a GPU is available and enables memory growth to optimize training performance.
</li><br>

<li><b>Load and Preprocess the Fashion MNIST Dataset : </b><ul>
<li>Loads the Fashion MNIST dataset, which consists of 70,000 images (60,000 for training, 10,000 for testing).</li>
<li>Normalizes pixel values to the range [0,1] to speed up training.</li>
<li>Reshapes the images from (28,28) to (28,28,1) to match CNN input format (grayscale images with 1 channel).</li>
</ul>
</li><br>

<li><b>Defining a Function to Create the CNN Model:</b><ul>a. Creates a CNN model with:
<li>Two convolutional layers (Conv2D) with ReLU activation.</li>
<li>Two max pooling layers (MaxPooling2D) to reduce feature map size.</li>
<li>A fully connected layer (Dense) with 128 neurons.</li>
<li>An output layer (Dense) with 10 neurons and softmax activation for classification.</li>
</ul>
b. Uses Adam optimizer and sparse categorical cross-entropy loss.

</li>
<br>
<li><b>Define a Function to Train and Evaluate the Model</b>
<ul>
    <li>Trains the model for 10 epochs.</li>
    <li>Evaluates the model’s final accuracy and loss</strong>.</li>
    <li>Plots accuracy and loss curves for training and validation sets.</li>
</ul>
</li><br>
<li><b>Train the Model with Different Hyperparameters</b></li>
<ul>
    <li>Effect of Filter Size: Tests two different filter sizes: 3×3 and 5×5.</li>
    <li>Effect of Regularization: Tests L2 regularization (λ=0.001) and no regularization.</li>
    <li>Effect of Batch Size: Trains the model using batch sizes of 32 and 64.</li>
    <li>Effect of Optimizer: Tests Adam and SGD (Stochastic Gradient Descent)optimizers.</li>
</ul>
<hr>
<h3>My Comments</h3>
<p>1. As per the accuracy result the following observations can be made :-</p>

<ul>
<li>Filter Size : Filter Size of 3 is more effective than 5.</li>
<li>Regularization : Applying no regularization is better than the L2 one.</li>
<li>Batch Size : Batch Size of 32 give better accuracy that Batch Size of 64.</li>
<li> Optimizer : Adam optimizer gives better accuracy than Stochastic Gradient Descent (SGD).
</ul>
<p>2. Apparently in this expriment the difference in accuracy is not very significant when changing the Filter Size, Regularization and Batch Size (i.e. less than 1 percent) but when we change the Optimizer to Adam the change in accuracy is most visible (i.e. ~3 percent). 

<p>3. The overall execution time to test the model against all the hyperparameters take about 10 minutes with using Nvidia RTX 4050 Graphics Card with 6gb of Vram. 













