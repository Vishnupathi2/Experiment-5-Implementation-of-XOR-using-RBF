# Experiment-5-Implementation-of-XOR-using-RBF

## AIM :

To classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.
  
## EQUIPMENTS REQUIRED :

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT :

Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
<img width="541" alt="image" src="https://user-images.githubusercontent.com/112920679/201299438-5d1926f9-25e9-4f20-b392-1c112880ef56.png">

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below
<img width="246" alt="image" src="https://user-images.githubusercontent.com/112920679/201299568-d9398233-71d8-41b3-8b08-a39d5b95e3f1.png">

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.


A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.

<img width="261" alt="image" src="https://user-images.githubusercontent.com/112920679/201300944-5510d7f4-ea0f-45ec-875d-87f463927e9d.png">

The RBF of hidden neuron as gaussian function 

<img width="206" alt="image" src="https://user-images.githubusercontent.com/112920679/201302321-a09f72e9-2352-4f88-838c-3324f6c5f57e.png">


## ALGORITHM :

### STEP 1 : 

Import the necessary libraries of python.

### STEP 2 :

In the function of end_to_end, calculate the similarity between the inputs and the peaks. Then, using the equation Aw = Y in matrix form. Each row of A consists of (shape: (4, 2)).

### STEP 3 :

In index[0] of the matrix there is a similarity of a point with peak1. In index[1] of the matrix there is a similarity of another point with peak2. In index[2] of the matrix the Bias input is given (1).

### STEP 4 :

Y is the output associated with the input of (shape: (4, )). W is calculated using the same equation we use to solve the linear regression using a closed solution ((i.e) normal equation).

### STEP 5 :
This neural network is constructed using the architecture of 2-2-1;


## PROGRAM :

### DEVELOPED BY : vishnupathi A

### REG NO : 212221223004

### Program to classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.

```
import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark)**2)
    
def end_to_end(X1, X2, ys, mu1, mu2):
    from_1 = [gaussian_rbf(i, mu1) for i in zip(X1, X2)]
    from_2 = [gaussian_rbf(i, mu2) for i in zip(X1, X2)]

    # plotting:
    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.scatter((x1[0], x1[3]), (x2[0], x2[3]),label="Class_0")
    plt.scatter((x1[1], x1[2]), (x2[1], x2[2]),label="Class_1")
    plt.xlabel("$ X1 $", fontsize=15)
    plt.ylabel("$ X2 $", fontsize=15)
    plt.title("X-OR : Linearly Inseparable", fontsize=15)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(from_1[0], from_2[0], label="Class_0")
    plt.scatter(from_1[1], from_2[1], label="Class_1")
    plt.scatter(from_1[2], from_2[2], label="Class_1")
    plt.scatter(from_1[3], from_2[3], label="Class_0")
    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate("Seperating Hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(f"$ mu1 $ : {(mu1)}", fontsize=15)
    plt.ylabel(f"$ mu2 $: {(mu2)}", fontsize=15)
    plt.title(" Transformed Inputs : Linearly Seperable ", fontsize=15)
    plt.legend()

    A = []

    for i, j in zip(from_1, from_2):
        temp = []
        temp.append(i)
        temp.append(j)
        temp.append(1)
        A.append(temp)

    A = np.array(A)
    W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(ys)

    print(np.round(A.dot(W)))

    print(ys)

    print(f"Weights: {W}")

    return W
    
def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(np.array(point), mu1)
    gaussian_rbf_1 = gaussian_rbf(np.array(point), mu2)
    A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
    return np.round(A.dot(weights))
    
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])

mu1 = np.array([0, 1])
mu2 = np.array([1, 0])

w = end_to_end(x1, x2, ys, mu1, mu2)

print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}")

print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), w)}")

print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")

print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")
```


## OUTPUT :

### X-OR : Linearly Inseparable v/s Transformed Inputs : Linearly Seperable :


![image](https://github.com/Abrinnisha6/Experiment-5-Implementation-of-XOR-using-RBF/assets/118889454/526101cf-c190-486c-9a25-e561e6b89060)


![image](https://github.com/Abrinnisha6/Experiment-5-Implementation-of-XOR-using-RBF/assets/118889454/04261a44-0f5d-456c-b9e9-21fcc49115b2)


### Printing the given input and the corresponding predicted values :

![image](https://github.com/Abrinnisha6/Experiment-5-Implementation-of-XOR-using-RBF/assets/118889454/d546e621-35cf-4d75-a8e3-95fb94149f93)


## RESULT :

Thus, the implementation of X-OR using the radial basis function (RBF) to classify the Binary input patterns is executted successfully .
