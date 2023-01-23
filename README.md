# Kervolutional Neural Netwroks

Convolutional neural networks are the current state of the art solution for the problem of image classification.A novel approach in the traditional 
successful CNN is proposed,that uses a projection to higher dimensions using the kernel trick.

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
The Google colab notebook, with theoritical detail and cod eimplemenattion can be found in the kervolutionalNN.ipynb page. 
