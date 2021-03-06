# Intro to NN
## Classification Problem

Lets build a linear regression line.

For the example dataset university acceptance

GRADES = X1
TEST_SCORES = X2

## Linear Boundries

Boundry Line:

2x1 + x2 + 18 = 0

if postive: accept
else: reject

Building a generic boundry line:


w1x1 + w2x2 + b = 0

Wx + b = 0
W = (w1, w2)
x = (x1, x2)

y = label: 0 or 1


question:


Now that you know the equation for the line (2x1 + x2 - 18=0), and similarly the “score” (2x1 + x2 - 18), what is the score of the student who got 7 in the test and 6 for grades?

ANSWER: 2



## Higher Dimensions Boundries

More variables in the dataset

GRADES = X1
TEST_SCORES = X2
CLASS_RANK = x3
...

Boundry Plane:
w1x1 + w2x2 + w3x3 + b = 0

WX + b = 0

N-Dimensional Space



question:
Given the table in the video above, what would the dimensions be for input features (x), the weights (W), and the bias (b) to satisfy (Wx + b)?

W: (1 * n), x: (n * 1), b: (1 * 1)



## Lets build a Perceptron

Perceptron is encoding the equation into a small graph

given inputs (test=7, grade=6) -> pass through linear boundry function
 and return if 1/0 for true/false


General form

w/ bias a param

x1 -> w1 ->
x2 -> w2 ->
x3 -> w3 ->    (Wx +b  = SUM(WiXi + b)) -> STEP(True > 0, else False)
....
xn -> wn ->
1 -> b ->

(optional can include bias a datapoint vs storing it inside the node)

datapoints  -->   Linear Function --> Step Function --> Output


## Perceptrons as logical Operators

In this lesson, we'll see one of the many great applications of perceptrons. As logical operators! You'll have the chance to create the perceptrons for the most common of these, the AND, OR, and NOT operators. And then, we'll see what to do about the elusive XOR operator. Let's dive in!


```

import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 1
weight2 = 1
bias = -2

# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
```


## Perception Tricks

GOAL: Splitting Data


To get a line closer to a point, you can adjust that using MATH!
Lets document that MATH!

Line: 3x1 + 4x2 - 10 = 0

Point( 4,5)(1-bias) is misclassfied, should be BELOW the line
```
FULL MOVE:
 3     4    -10
-4    -5    - 1
---------------
-1    -1    -11
```
BUt that is bad cuz it moves to much.....

SO We are going to modify the cordinates to adjust
We could just FULL HAM and just substract the point for the weight...
but that would over correct


SO  We need to balance the SLOW adjustment of the line but in a smoother format.
SO we introduce the Learning RATE
IE a modifier on how much we adjust the function with respect to new data

Line: 3x1 + 4x2 - 10 = 0
`LEARNING_RATE = 0.1`


```
FULL MOVE:
 3     4     -10
-0.4  -0.5   -0.1
-------------------
2.6   3.5    -10.1
```


## Now lets build the Perceptron: Coding

Coding the Perceptron Algorithm
Time to code! In this quiz, you'll have the chance to implement the perceptron algorithm to separate the following data (given in the file data.csv).

Recall that the perceptron step works as follows. For a point with coordinates (p,q) , label y, and prediction given by the equation `​y​^​​ =step(w​1​​ x​1​​ +w​2​​ x​2​​+b)`

- If the point is correctly classified, do nothing.
- If the point is classified positive, but it has a negative label, subtract αp,αq, and α from w1​ ,w2​ , and b respectively.
- If the point is classified negative, but it has a positive label, add αp,αq, and α to w​1​​ ,w​2​​ , and b respectively.

```
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        pred = prediction(X[i], W, b)
        if y[i] - pred == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i] - pred == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate

    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
```


## NON-Linear Regions

If you cant fit a linear line... you got a problem and need to use a different thing

## Error Functions

Check distance from success and move in that lowest place.

## Log-Loss Error function


Essentially, look for the lowest loss and go to that point

Needs to be Continous vs discrete.

DISCRETE -> Step function (YES/NO)
Continious -> Numeric range (90 probability of YES)

Replacing output of the function (Wx + b = y^) then pass through sigmoid (x) = 1/(1+e^-x)
You get a probablity distribution

Label being postive or negative

## How to handle mutliclass

Essentailly you need to convert the regression score into a probablity distubtion
You can do this by taking the scores and passing them through a softmax function
to get their adjust probablity distribution across the e^mean.



```
# softmax
import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    base = sum(np.exp(L))
    return [np.exp(i) / base for i in L]
    ```


Given a probablity of p(Duck) == 0.67, p(beaver) == 0.24, P(walrus) == 0.09

and a Linear score of Duck = 2, beaver = 1 ,  walrus = 0

then pass through softmax for WIN (aka convert to probablities)


### One-hot encoding

yea, one hot that shit


## Maximum Likelihood


Pick the models which give the BEST probablity

How to do this...

Compute the TRUE probablity for that point, then muliplie together to get a max probablity

ie chart
```

# compute the probablity and select the actual prob to get likehood

   p(b) p(r) actual likehood
p1 0.9   0.1   r   0.1
p2 0.3  0.7    r   0.7
p3 0.6  0.4    b   0.6
p4 0.2  0.8    b   0.2


there for the likehood of this line is:

0.1 * 0.7 * 0.6 * 0.2 = 0.0084

```


SO multiples are bAD... so lets switch to SUM...


We can do this by taking a logarthim

```
# bad model

ln(0.1)+ ln(0.7) + ln(0.6) + ln(0.2) = -4.8
-0.51 -1.61 -2.3 - 0.36 = -4.8


# good model

ln(0.7)+ ln(0.9) + ln(0.8) + ln(0.6) = -1.2
-0.36 - 0.1 - 0.22 - 0.51 = -1.2


```
Closer to 0 is better since it shows there Error is closer to truth.

## Cross Entropy


computing the loss at each point and then you can minimize the function of the error


###  Cross Entropy Formula

cross-entropy = -SUM (y1 * ln(p1)) + (1 -y1)* ln(1 - p1)

```
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    cross = 0
    for y, p in zip(Y, P):
        cross -= y*np.log(p) + (1-y) *np.log(1-p)
    return cross

  ```

###  Cross Entropy Multiclass

Compute the probablity then cross entropy for that class... duh!


cross-entropy = -SUM(n:i SUM(m:j) yij * ln(pij) )


## Logistic Regression


Now, we're finally ready for one of the most popular and useful algorithms in Machine Learning, and the building block of all that constitutes Deep Learning. The Logistic Regression Algorithm. And it basically goes like this:

- Take your data
- Pick a random model
- Calculate the error
- Minimize the error, and obtain a better model
- Enjoy!


Calculating the Error Function
Let's dive into the details. The next video will show you how to calculate an error function.

E(W, b) = - 1/m SUM(m -> i=1) (1-yi)* ln(1 - sigmoid(Wx(i) + b)) + (yi)* ln(sigmoid(Wx(i) + b))

### How to minimize the error function


GOAL: Min Error Function

process -> Gradient Descent


# Gradient Descent
In the last few videos, we learned that in order to minimize the error function, we need to take some derivatives. So let's get our hands dirty and actually compute the derivative of the error function. The first thing to notice is that the sigmoid function has a really nice derivative. Namely,



`σ′(x)=σ(x)(1−σ(x))`


This actually tells us something very important.
For a point with coordinates (x​1​​ ,…,x​n​​ ), label y, and prediction ​y​^​​ , the gradient of the error function at that point is
`(−(y−​y​^​​ )x​1​​ ,⋯,−(y−​y​^​​ )x​n​​ ,−(y−​y​^​​ ))`.

In summary, the gradient is `∇E=−(y−​y​^​​ )(x​1​​ ,…,x​n​​ ,1)`.

If you think about it, this is fascinating. The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. What significance does this have?

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

# Gradient Descent Step

Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function at every point, then this updates the weights in the following way:

`w​i​′​​ ←w​i​​ −α[−(y−​y​^​​ )x​i​​ ]`,which is equivalent to `w​i​′​​ ←w​i​​ +α(y−​y​^​​ )x​i​​` .Similarly, it updates the bias in the following way:

`b​′​​ ←b+α(y−​y​^​​ )`,
Note: Since we've taken the average of the errors, the term we are adding should be
​`m1​ ⋅α` instead of α, but as α is a constant, then in order to simplify calculations, we'll just take `m​​1​​ ⋅α` to be our learning rate, and abuse the notation by just calling it α.
NEXT


# Gradient Decent sudo code


Step 1: Start with random weights:
`w1, .... wn, b`

step 2: calculate the error for every point,
	(smaller for correct points, big for incorrect points)
	2.1 for i=1.... n
		a - learning rate
		2.1.1 update w' <-- w - a (y^ - y)xi
		2.1.2 update b' <-- b - a (y^ - y)


step 3: repeat until process untill error is small


## Neural Network Architecture

How to create non-linear models...

combine 2 linear models into a non-linear model

Model1 + Model2 = newModel


ie
model1score * 1bias + model2score * 2bias = combined score

Now combining with the perceptron concept you can combine the 2 linear perceptrons into
a NN by matching up the lineaity and passing it to an output.



# Feedforward: Training the NN

Pass the data through each node and compute the next layer


# Backpropagation

Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as backpropagation. In a nutshell, backpropagation will consist of:

- Doing a feedforward operation.
- Comparing the output of the model with the desired output.
- Calculating the error.
- Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
- Use this to update the weights, and get a better model.
- Continue this until we have a model that is good.

Sounds more complicated than what it actually is. Let's take a look in the next few videos. The first video, will show us a conceptual interpretation of what backpropagation is.


reminder that the chain rule applys here
in order to get the deriavitve of the error with respect to the label, then you take the partial deriviatves with the repspect to each of the peices.

Remember the sigmoid function has a really simple derivative... `sig'(x) = sig(x)(1-sig(x))`

