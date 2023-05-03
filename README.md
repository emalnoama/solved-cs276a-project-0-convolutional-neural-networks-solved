Download Link: https://assignmentchef.com/product/solved-cs276a-project-0-convolutional-neural-networks-solved
<br>
<h1>1        Objective</h1>

The ImageNet challenge initiated by Fei-Fei Li (2010) has been traditionally approached with image analysis algorithms such as SIFT with mitigated results until the late 90s. The advent of Deep Learning dawned with a breakthrough in performance which was gained by neural networks. Inspired by Yann LeCun et al. (1998) LeNet-5 model, the first deep learning model, published by Alex Krizhevsky et al. (2012) drew attention to the public by getting a top-5 error rate of 15.3% outperforming the previous best one with an accuracy of 26.2% using a SIFT model. This model, the so-called ’AlexNet’, is what can be considered today as a simple architecture with five consecutive convolutional filters, max-pool layers, and three fully-connected layers.

This project is designed to provide you with first-hand experience on training a typical Convolutional Neural Network (ConvNet) model in a discriminative classification task. The model will be trained by Stochastic Gradient Descent, which is arguably the canonical optimization algorithm in Deep Learning.

<h1>2        Model</h1>

The ConvNet is a specific artificial neural network structure inspired by biological visual cortex and tailored for computer vision tasks. The ConvNet is a discriminative classifier, defined as a conditional (posterior) probability.

(1)

where <em>c </em>∈ {1<em>,</em>2<em>,…,C </em>= 10} is the class label of an input image <em>I</em>, and <em>f<sub>c</sub></em>(<em>I</em>;<em>w</em>) is the scoring function for each category, and computed by a series of operations in the ConvNet structure. Figure 1 displays an structure (architecture) of the ConvNet.

<ol>

 <li><strong>Convolution</strong>. The convolution is the core building block of a ConvNet and consists of a set of learnable filters. Every filter is a small receptive field. For example, a typical filter on the first layer of a ConvNet might have size 5x5x3 (i.e., 5 pixels width and height, and 3 color channels). Figure 2 illustrates the filter convolution.</li>

 <li><strong>Pooling</strong>. Pooling is a form of non-linear down-sampling. Max-pooling is the most common. It partitions the input image into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum.</li>

 <li><strong>Relu</strong>. Relu is non-linear activation function <em>f</em>(<em>x</em>) = <em>max</em>(0<em>,x</em>). It increases the nonlinear properties of the decision function and of the overall network without affecting the receptive fields of the convolution layer.</li>

</ol>

Figure 1: A ConvNet consists of multiple layers of filtering and sub-sampling operations for bottom-up feature extraction, resulting in multiple layers of feature maps and their sub-sampled versions. The top layer features are used for classification via multi-nomial logistic regression

Figure 2: Filter convolution. Each node denotes a 3-channel filter at specific position. Each filter spatially convolves the image.

LeNet shown in Table 2 is a typical structure design taylored for the CIFAR-10 dataset. The Block1 has 32 filters and the filter size is 5x5x3. The Block5 is a logistic regression layer, in which the number of filters is the number of classes and the filter size should be the same as the output from Block4. The CIFAR-10 dataset consists of 60<em>,</em>000 32×32 color images in 10 classes, with 6<em>,</em>000 images per class. There are 50<em>,</em>000 training images and 10<em>,</em>000 testing images. Figure 3 shows example images from 10 categories. The objective of classification is to predict the category of each image.

<table width="551">

 <tbody>

  <tr>

   <td width="105">Block1</td>

   <td width="112">Block2</td>

   <td width="112">Block3</td>

   <td width="112">Block4</td>

   <td width="112">Block5</td>

  </tr>

  <tr>

   <td width="105">Conv 5x5x3x32</td>

   <td width="112">Conv 5x5x32x32</td>

   <td width="112">Conv 5x5x32x64</td>

   <td width="112">Conv 4x4x64x64</td>

   <td width="112">Conv 1x1x64x10</td>

  </tr>

  <tr>

   <td width="105">Pooling 3×3</td>

   <td width="112">Relu</td>

   <td width="112">Relu</td>

   <td width="112">Relu</td>

   <td width="112">Softmax</td>

  </tr>

  <tr>

   <td width="105">Relu</td>

   <td width="112">Pooling 3×3</td>

   <td width="112">Pooling 3×3</td>

   <td width="112"> </td>

   <td width="112"> </td>

  </tr>

 </tbody>

</table>

Figure 3: CIFAR-10 Dataset. CIFAR-10 contains 60,000 images in 10 classes.

<h1>3        Assignment</h1>

In this project, you will train variations of a LeNet in TensorFlow or PyTorch:

<ol>

 <li>Optimization of LeNet.

  <ul>

   <li>Complete the functions <em>flatten</em>(), <em>convnet init</em>(), <em>convnet forward</em>().</li>

   <li>Adjust parameters <em>learning rate</em>, <em>epochs </em>such that test-accuracy <em>&gt; </em>70%.</li>

   <li>Plot the training loss and test accuracy over epochs in two Figures.</li>

  </ul></li>

 <li>Alteration of LeNet.

  <ul>

   <li>Keep the Block5, learn a ConvNet only with:

    <ol>

     <li></li>

     <li>Block1 and Block2.</li>

    </ol></li>

  </ul></li>

</ol>

<ul>

 <li>Block1, Block2 and Block3.</li>

</ul>

<ul>

 <li>Compare the final test accuraries for (i., ii., iii.) in a Table. (<strong>Hint</strong>: You need to change the filter size in Block5 to match the output from previous block.)</li>

</ul>

<ol start="3">

 <li>Visualization of filters and activations.

  <ul>

   <li>Plot the learned 32 filters of the first convoluational layer in LeNet.</li>

   <li>Plot the filter response maps for a given sample image of CIFAR-10.</li>

  </ul></li>

</ol>


