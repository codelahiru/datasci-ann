Perceptron Basics
Structure:

A perceptron consists of input features, weights associated with each input, a bias term, and an activation function (typically a step function).
The inputs are fed into the perceptron, and each input is multiplied by its corresponding weight. The bias term is added to the weighted sum of the inputs.
Activation Function:

The perceptron uses an activation function to determine the output. A common choice is the step function, which outputs 1 if the weighted sum is above a certain threshold and 0 otherwise.
Mathematically:
𝑜
𝑢
𝑡
𝑝
𝑢
𝑡
=
{
1
if 
∑
(
𝑖
𝑛
𝑝
𝑢
𝑡
×
𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
)
+
𝑏
𝑖
𝑎
𝑠
>
0
0
otherwise
output={ 
1
0
​
  
if ∑(input×weight)+bias>0
otherwise
​
 
