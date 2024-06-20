---
layout: post
title: "Intuition for Fundamentals of Machine Learning: Backpropagation"
date: 2024-06-10
categories: datascience
---
This is really meant for me to look back on. To really reinforce and remember fundamental concepts of machine learning.
## Forward Propagation
Given the following variables:
- \( x \): Input to the neuron
- \( W \): Weight
- \( b \): Bias
- \( y$$_{true}$$ \): True output value

### 1. Linear Combination \( z \)
$${\Huge z = Wx + b}$$

### 2. Activation \( a \)
The activation function (sigmoid in this case) applied to \( z \):
 $${\Huge a = \sigma(z) = \frac{1}{1 + e^{-z}}}$$

### 3. Output \( y \)
For simplicity, we use the activation as the output:
$${\Huge y = a}$$

### 4. Loss \( L \)
The loss function (Mean Squared Error):
$${\Huge L = \frac{1}{2}(y - y_{\text{true}})^2}$$

## Backpropation
- Backpropagation is calculated using the chain rule. The chain rule is for composite functions.
- $${\Huge \frac{d}{dx}[f(g(x))] = f^`(g(x)) * g^`(x)}$$

### 4a. Loss function as a composite function
- $${\Huge L = \frac{1}{2}(y - y_{\text{true}})^2}$$
- $${\Huge L = \frac{1}{2}(a - y_{\text{true}})^2}$$
- $${\Huge L = \frac{1}{2}(\sigma(z) - y_{\text{true}})^2}$$
- $${\Huge L = \frac{1}{2}(\sigma(Wx + b) - y_{\text{true}})^2}$$
### 5. Showing the derivative as a composite function
#### Derivative of Loss Function with respect to W
#### https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/a/chain-rule-review
- We need how the loss would change as W changes hence we need $${\frac{dL}{dW}}$$.
- $${\Huge \frac{dL}{dW} = \color{green}{\frac{2}{2}(\sigma(Wx+b) - Y_{true})} * \color{red}{1} * \color{orange}{\sigma(Wx+b) *(1-\sigma(Wx+b))} * \color{purple}{x}}$$
- $${\Huge \frac{dL}{dW} = \color{green}{\frac{dL}{dy}} * \color{red}{\frac{dy}{da}} *\color{orange}{\frac{da}{dz}} *\color{purple}{\frac{dz}{dW}}}$$
