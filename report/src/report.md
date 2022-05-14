# Rapid Implementation of Hardware Neural Networks Powered by Verython

## Project Introduction

## Introduction to Notation
For some array $A$, let $A:[d1,\dots,d_i,\dots,d_n]$ indicate that that $A$ is $n$-dimensional and each dimension $1\leq i\leq n$ has a length of $d_i$. For example,
$$A:[2, 3]=\begin{bmatrix} a_{1, 1} & a_{1,2} & a_{1,3} \\ a_{2,1} & a_{2,2} & a_{2,3} \end{bmatrix}.$$

## High Level Design

## Program/Hardware Design

### Model Implementation
In this section, we will dive deep into the math behind each of the layers in the model and how we rederived the transformation functions of each layer to fit on the FPGA. 

As a reminder, our model ingests some image of a hand-drawn digit and outputs a number between 0 and 9 which corresponds to the prediction of the model. The image is a 28 by 28 2D array of values between 0 and 255, or rather, some input image $img$ can be represented as
$$img=\begin{bmatrix} 
  p_{1,1} & \dots & p_{1,j} & \dots & p_{1,28} \\ 
  \vdots & \ddots & & & \vdots \\
  p_{i, 1} & \dots & p_{i, j} & \dots & p_{i, 28} \\
  \vdots & & & \ddots & \vdots \\
  p_{28, 1} & \dots & p_{28, j} & \dots & p_{28, 28}
\end{bmatrix}$$
such that $0\leq p_{i,j}\leq255$. 

Moreover, we can think of our model $M$ as a function that maps some input $img$ to some output prediction $pred$, or rather, the inference of our model can be represented by 
$$M(img)=pred$$
for some output prediction $0\leq pred\leq9$. 

Furtheremore, we can also express each of the layers in our model as a a function. Let $L=\{L_1,\dots,L_i,\dots,L_4\} be the layers in our model. Then we can represent each layer $L_i$ as some function 
$$L_i(V_i^{in})=V_i^{out},$$
where $V_{in}$ and $V_{out}$ are matrices whose shapes are defined by $L_i$. 

With this new notation, we can redfine $M$ in terms of its layers:
\begin{align*}
  M(img) & =L_4(L_3(L_2(L_1(img)))) \\
  & =pred.
\end{align*}
Thus, a model's prediction is simply the output of its cascaded layer functions.

Let's now take a look at each of the layers and how we can rederive their functions to be built on the FPGA.