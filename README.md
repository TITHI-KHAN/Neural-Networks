# Neural Network

• Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. Specific applications of AI include expert systems, natural language processing, speech recognition, and machine vision.

• Machine Learning is the study that uses statistical methods to enable machines to improve with experience.

• Deep learning is a subset of machine learning, which is a subset of AI. Artificial intelligence is any computer program that does something smart. Deep Learning is the study that makes use of Neural Networks (similar to neurons present in the human brain) to imitate functionality just like a human brain.

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/1b717542-214a-4b92-a28b-f4fbb5b0aa4e)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/e49ee723-fb72-4fa6-8db3-65610e59f698)

CS -> Cognitive System (A part of Linguistics). 

DL + Linguistics + CS -> NLP.

Ex: BARD, ChatGPT.

NLP-powered machine translation helps us to access accurate and reliable translations of foreign texts. Natural language processing is also helping to optimize the process of sentiment analysis. Natural language processing-powered algorithms can understand the meaning behind a text. Voice assistant, Alexa using Natural Language Processing provides a variety of services using artificial intelligence systems equipped through the user's voice commands. NLP Top projects-

▪ Email filters

▪ Voice Assistant

▪ Amazon Alexa

▪ Google Translator

▪ Voice Translator

▪ Text Analysis

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6d92cb8e-98c0-4954-b961-19e9ac0b6279)

In the architecture of a Neuron, Dendrite -> Input and Soma -> Cell Body. Here, all the calculation takes place in Soma and the result will go through the Axon. This information will be received by another neuron (in the axon terminals). For the first neuron, it will be the output but for the second neuron, it will be the input. Like this way, a network is created from lots of neurons.

**ANN**: 

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/d8900493-c68f-4cdf-b002-371be7d41d15)

In Forward Propagation, input * weight = result and the result gets forwarded.

Here, x1 is a different feature. For this, x1 has a different weight. The same goes for x2, x3, and xn as well.

Sum =  where w0 -> value of bias.

w1 -> initially assigned value (random weight).

While doing Back Propagation, the value gets updated. 

In Forward Propagation, it goes from Sum to Activation Function. Activation Function -> result will go to which class? yes or no? (Result = x1 w1 + x2 w2 + ..... xn wn + w0)

There are lots of activation function : sigmoid, soft max, Tanh, ReLU, Leaky ReLU, PReLU, ELU, Swish.

For **Binary Classification**, the best activation function is **Sigmoid**.

1/1+e^(-x) [e has a value]

Suppose, e=2.74 and x=0.23 (x1 w1 + x2 w2 + ..... xn wn + w0=0.23), then 1/1+e^(-x) = 0.93 

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/ffbf1fa4-9eb1-41d0-ad1f-2a18b71ed78a)

First, we must talk about neurons, the basic unit of a neural network. A neuron takes inputs, does some math with them and produces one output. Here’s what a 2‐input neuron looks like:

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6aaff771-7c71-4015-a2e2-96c16e505c1f)

In neural network training, the weights are typically initialized randomly and then adjusted during the training process through backpropagation. Backpropagation is an iterative algorithm that updates the weights in the network based on the error between the predicted output and the actual output. By minimizing this error, the network can learn to make more accurate predictions.

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/13c39b76-c072-4afe-b9d8-77b5e2b0321f)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/30dbcf8b-dd0c-41ca-a9c1-8f42948fe613)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/716166ce-9e65-496b-8a38-239d085ffb9e)

Specifying the epochs=10 means that the neural network will train on the entire training dataset for 10 iterations. During each epoch, the neural network will update its weights multiple times using backpropagation and stochastic gradient descent (or other optimization algorithms) until it has seen all the training examples. The number of weight updates during an epoch depends on the batch_size, which is another hyperparameter that determines how many samples are used to update the weights in each iteration.

**For example**, if we have a training dataset of 1000 samples and set the batch size to 100, the neural network will update its weights 10 times during an epoch (since 1000/100 = 10). During each weight update, the neural network will calculate the gradient of the loss function with respect to the weights and use this gradient to adjust the weights in the direction that reduces the loss.

After 10 epochs, the neural network will have updated its weights 10 times on the entire training dataset and hopefully learned to make accurate predictions on new data.

**What is the batch size in neural network training?** 

The batch size is a hyperparameter that specifies the number of training examples used in one iteration of the optimization algorithm. The training examples are divided into small groups or batches, and the optimization algorithm updates. The batch size is a number of samples processed before the model is updated.

The chain rule is a fundamental rule of calculus that is used to calculate the derivative of a composition of functions. In the context of neural networks, the chain rule is a key mathematical tool for calculating gradients of the loss function with respect to the weights of the network during the backpropagation algorithm.

In simple terms, the chain rule states that if a function y is a composition of two functions u and v, where y = f(u(v(x))), then the derivative of y with respect to x can be expressed as the product of the derivatives of u and v with respect to x, Mathematically, dy/dx = (du/dx) * (dv/dx).

In the context of neural networks, the chain rule is used to calculate the gradients of the loss function with respect to the weights in each layer of the network. During backpropagation, the gradients are calculated by recursively applying the chain rule from the output layer back to the input layer.

Specifically, the gradients of the loss with respect to the output of a layer are multiplied by the gradients of the layer's activation function and the weights connecting the layer to the previous layer, to obtain the gradients of the loss with respect to the inputs of the layer. These gradients are then used to update the weights of the layer using an optimization algorithm, such as stochastic gradient descent.

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/adc72d75-63cd-4139-86e8-ab8c25b96138)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/d1ff176b-bf91-487a-8957-8b19dcdb97c9)

# Problem

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/5bad9b8f-0c45-48b8-8a85-7b8ebd50e66d)

**Solution:**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/070bd000-1356-47af-a48d-03d3727601ea)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/72496079-8ddc-4358-b017-c2e14be318b0)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6afe1ce3-dd1c-4288-a43a-8440fc2508c6)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6a932f30-bcfe-4f7e-bed1-3c7eb392bb9b)

# Convolutional Neural Network (CNN)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/9936307b-0b0d-4b10-b2c8-045fce4823a1)

▪ Images recognition

▪ Images classifications

▪ Objects detections

▪ Decoding facial recognition

▪ Understanding climate

▪ Driverless cars

▪ Human genome mapping

▪ Predicting earthquakes

▪ Natural disasters

**How many layers does CNN have?**

There are three types of layers in a convolutional neural network: Convolutional layer, Pooling layer, and Fully connected layer. Each of these layers has different parameters that can be optimized and performs a different task on the input data.

**Images: Grayscale**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/178c4ea2-a603-444a-b15d-2940b3031f69)

**Images: RGB**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/51cdee5a-f3f6-4b13-97d1-fec2c325abc8)

**Convolutional Neural Network (CNN):**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/2df85750-6117-4bed-86c4-b057f483ca58)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/c0bb3385-19e6-4fd0-92ab-4ce494e3968b)

**Convolution Layer — The Kernel**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/56b5db08-599f-472c-a7e6-909b1754b89b)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/4ce84152-5a00-476a-8aef-4ae682a9c6db)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/0ccbdd14-197a-4c9c-a95a-8876129a5983)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/5c363009-b572-409a-83e5-afb034f45dfa)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/bfce1d8f-ce46-4d2e-93a7-deb60c5a5b0b)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6606bfde-cb1c-442b-b0a0-e486a49d74bc)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/fa3bd0f7-6ae0-4eb6-b517-13e5cccfb78a)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/2ef8ad48-1f6b-4197-b669-4789e940c852)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/f4953e1d-76fa-42d9-b41a-dcb90a9e2319)

**Convolution Layer — The Kernel**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/110c7253-0c7d-4b65-b035-7a2f1234b3d9)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/79229e8e-a780-4fcc-a569-e7b3ba591b84)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/f63df293-6fed-473a-9e9b-9c1295300ebc)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/c43e47e3-4b22-433b-95c3-0dc5e71b5e03)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/a233c913-4523-49d4-a309-05300bbd324e)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/68173b21-f9a3-42ae-b866-23847876cbce)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/df7aa467-7fd1-44ef-8b81-0e46ef4208c1)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/f78564a4-4932-401f-90ed-37e70fc5ec29)

# Activation function: Rectified Linear Units (ReLU)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/a6d45cc3-806f-4932-9e8d-6fa96a6a36a2)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/197abbab-5dea-4e13-9ce2-9e640673e33b)

# Pooling Layer

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/9ed6dc34-8342-4308-a251-798e9edd1c68)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/904dd5f1-011b-454e-963e-7b0b6fdf900a)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/c8726f73-b548-457e-bc90-be2dedbdbecf)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/fa197342-4a7e-459f-be36-b0b9204e956b)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/4ee972f4-30b2-413a-9457-c8bbfc8cb4ce)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/ac818f5d-1cfc-4910-894f-e8c9ad7094f1)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/ab228577-3303-4972-a556-1a388d6f7f4d)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/2d469465-b7f2-4c37-8316-f08036bd6877)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/2afa409c-125f-4a28-b046-c757ebccaf99)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6657b18a-96d1-48a2-b7bc-a3ac8024d601)

# Padding

**What is padding in CNN?**

Padding is a term relevant to convolutional neural networks as it refers to the amount of pixels added to an image when it is being processed by the kernel of a CNN. For example, if the padding in a CNN is set
to zero, then every pixel value that is added will be of value zero.

**Why does CNN use padding?**

Padding is simply a process of adding layers of zeros to our input images so as to avoid the problems mentioned above. This prevents shrinking as, if p = number of layers of zeros added to the border of
the image, then our (n x n) image becomes (n + 2p) x (n + 2p) image after padding.

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/40acad3a-00b1-42a6-b7ee-460d3e87bec7)

**Types of Padding:**

▪ **Valid Padding** : It implies no padding at all. The input image is left in its valid/unaltered shape. So, 

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/b56d5b18-14e7-4dc7-b325-f05e0c87e593)

▪ **Same Padding** : In this case, we add ‘p’ padding layers such that the output image has the same dimensions as the input image. So,

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/18ecc9eb-aaab-4623-b972-b34a2d43bd08)

# Data Augmentation in Deep Learning

**How do I get more data, if I don’t have “more data”?**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/b9a1fb85-be82-4711-8588-fd618a39957f)

1. **Zoom in/out**
   
2. **Rotation**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/4c3636da-4b1d-4dc1-a134-d5e8a91800ee)

3. **Flipping**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/39d6e80b-983c-426a-b452-9a43c572c084)

4. **Adding Noise**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/1ab9d2b7-9f70-4b08-a301-1c1a6d7df576)

5. **Lighting Condition**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/8817dab2-13f6-4e3e-9e34-3a944c6d399e)

6. **Perspective transform**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/f76bdb22-8f4e-46a0-a1a8-21486286d9b3)

**Watch**: https://www.youtube.com/watch?v=p8e7dGY-Oko

