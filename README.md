# Kervolutional Neural Netwroks

Kervolutional Neural Networks (KNN) are a [novel approach](https://arxiv.org/pdf/1904.03955) to deep learning image classification that leverages kernel methods to enhance the performance of convolutional neural networks (CNNs). This technique allows for more flexibility in modeling complex patterns in data, particularly in tasks involving image recognition and classification.

### Key features
* **Kernel Methods**: KNN utilizes kernel functions to transform data into higher-dimensional spaces, enabling better separation of classes.
* **Convolutional Layers**: Similar to traditional CNNs, KNN incorporates convolutional layers to extract features from input data.
* **Flexibility**: The kernel approach allows KNN to adapt to various data distributions, making it suitable for diverse applications.

### Overview
This project demonstrates the implementation of KNN using PyTorch. The provided Colab notebook showcases how to build and train a Kervolutional Neural Network for image classification tasks and compare it's performance against traditional CNN architectures.

### Intuition
The simple convolution operation in 2d,has two equal sized windows , multiplied element-wise and sum up the individual results.
In the image below,the blue 3x3 corner in the left matrix,is convoluted with the 3x3 matrix in the middle to produce the result 51,in the right matrix
![1_xBkRA7cVyXGHIrtngV3qlg](https://user-images.githubusercontent.com/67536962/121771470-3c1d4f80-cb78-11eb-828f-217e70ecfe55.png)

The kervolution operation, on the other hand, performs the convolution operation, not by multipyling each element of the matrix with the corresponding matrix element of the kernel, but by multiplying two higher dimension representations of the aforementionded quantities. \
Let's say that $f(x)=x \otimes w$ is the convolution operation. The kervolution operation would be $f(x)=<\phi(x),\phi(w)>$, were $\phi$ is a higher dimensional representation of its input.By going into higher dimensions we can ,in general, capture more difficult relations and thus make more precise classifications.\
This computationally insufficient multiplication can be avoided by applying the kernel trick

$<\phi(x),\phi(w)> = \sum_{j}c_j(x^{T}w)^{j}=k(x,w)$

where $k$ is a kernel function. 

### Implementation
The Kervolutional Neural Network (KNN) is implemented using PyTorch, which provide a robust framework for building deep learning models. The architecture consists of multiple convolutional layers that apply kernel transformations to input data, followed by activation functions to introduce non-linearity. We utilize custom kernel functions to enhance the capacity of the model, allowing it to capture complex patterns in the data. The network is trained using standard backpropagation techniques, optimizing the loss function to adjust the weights of the kernels.

To facilitate experimentation, the implementation includes the loading of the tiny imagenet dataset from [Stanford university](https://cs231n.stanford.edu/), allowing users to evaluate the KNN's performance across different scenarios. The Colab notebook provides step-by-step instructions for setting up the environment, loading datasets, and training the models. Users can easily modify parameters such as kernel types and layer configurations to explore their effects on model performance. 