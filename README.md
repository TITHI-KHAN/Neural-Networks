# Neural Network

• Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. Specific applications of AI include expert systems, natural language processing, speech recognition, and machine vision.

• Machine Learning is the study that uses statistical methods to enable machines to improve with experience.

• Deep learning is a subset of machine learning, which is a subset of AI. Artificial intelligence is any computer program that does something smart. Deep Learning is the study that makes use of Neural Networks (similar to neurons present in the human brain) to imitate functionality just like a human brain.

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/1b717542-214a-4b92-a28b-f4fbb5b0aa4e)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/e49ee723-fb72-4fa6-8db3-65610e59f698)

NLP-powered machine translation helps us to access accurate and reliable translations of foreign texts. Natural language processing is also helping to optimize the process of sentiment analysis. Natural language processing-powered algorithms can understand the meaning behind a text. Voice assistant, Alexa using Natural Language Processing provides a variety of services using artificial intelligence systems equipped through the user's voice commands. NLP Top projects-

▪ Email filters

▪ Voice Assistant

▪ Amazon Alexa

▪ Google Translator

▪ Voice Translator

▪ Text Analysis

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6d92cb8e-98c0-4954-b961-19e9ac0b6279)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/d8900493-c68f-4cdf-b002-371be7d41d15)

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

**Problem:**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/5bad9b8f-0c45-48b8-8a85-7b8ebd50e66d)

**Solution:**

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/070bd000-1356-47af-a48d-03d3727601ea)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/72496079-8ddc-4358-b017-c2e14be318b0)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6afe1ce3-dd1c-4288-a43a-8440fc2508c6)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/6a932f30-bcfe-4f7e-bed1-3c7eb392bb9b)

# Convolutional Neural Network (CNN)

![image](https://github.com/TITHI-KHAN/Neural-Network/assets/65033964/5caf74f3-8372-4e6d-8a8e-c3e424757078)







