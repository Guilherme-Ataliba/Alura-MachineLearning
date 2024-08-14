# 🔵 Stratification

Train test split randomly distributes the data into training and testing sets. This could result in a uneven distribution, which will cause problems. **Stratification** is the processes used to address this problem, in which it divides the data in such a way that the different classes are divided equally in into training and testing sets.

- For example, the "vendido" feature is binary (0 or 1).

The stratification process ensures that the original proportion of each class in the original dataset is preserved in both the training and testing sets.

- For example, if the original data has 70% of class A and 30% of class B, stratification will ensure that both in training and testing sets they have approximately 70% of class A and 30% of class B.

  
  
# 🔵 Dummy Test

This is fundamental to understand how well your model actually is, thus you compare it to a "dummy model", a model that usually randomly guesses the answer. The model you've constructed should, at the lest, perform a little better than the dummy one.

- By default `sklearn`'s dummy classifier is already stratified. This means that it'll look at the training data distribution, count the occurrences of each class and then use those numbers to predict the outcome. 