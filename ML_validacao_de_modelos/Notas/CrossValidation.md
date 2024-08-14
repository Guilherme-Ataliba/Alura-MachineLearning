
# 🔵 Randomness

Usually you'll divide your data into **training and testing datasets** in a process of training some model in ML. The problem is this involves randomness (even with stratification), every time you run this algorithm you'd get a different outcome for your accuracy (or any other metric).

This is also the case for models that use randomness, with simple train test split it is not possible to know if the outcome you've gotten is caused by luck or it accurately expresses the model's capacity. 


# 🔵 K-fold Cross Validation
An ideia to deal with randomness caused by data variations is to use cross validation. 
1. In which case one divides the dataset into **$\boldsymbol{k}$ folds** (smaller datasets). 
2. Then, you train your model in $k-1$ folds and test it on the remainder set.
3. You then repeat this process $k$ times, until every fold has been used as testing set exactly once. 

![[Pasted image 20240813142022.png]]

Once every fold has been trained you should get $k$ measurements of quality (accuracy, scores, etc.). Then you may use different metrics to evaluate this output:

- **Average**
- **Confidence Interval**

It is important to notice that, as expected, there is a trade-off between computational time and how well your quality measurement is, as you change the value of K.

🔸 Something note worth is that there are a number of scientific papers that discuss the best value for **k**. The most accepted value currently ranges from 5 to 10. 

### 🔹 Types of Cross Validation
There are many different types of cross validation and ways to split your data. Each kind has its own applications and none is general. This means that you can't choose a single one and stick with it. You must understand your data and your problem so you can best choose the type of cross validation that best fits your situation.


### 🔹 Confidence Interval
Its an statistical fact that approximately 95% of the distribution is inside the interval
$$[\mu - 2\sigma, \mu+2\sigma]$$
- Where $\mu$ is the average and $\sigma$ the standard deviation. 

This interval is what we call confidence interval

### 🟢 In Python
```python
from sklearn.model_selection import cross_validate

results = cross_validate(modelo, X, y, cv = 10)

media = results["test_score"].mean()
std = results["test_score"].std()

```


# 🔷 Random Cross Validation
In the previous case, we have used the k-fold strategy to split our data. This is a deterministic processes, where in every case data will be split the same way. 

We could use different strategies to split the data in cross validation, and the most common is to simply shuffle the data before splitting into different strata. This ensures a less biased test. 


### 🟢 In Python
Previously we have informed to the parameter `cv` the number of splits to make. But this parameter actually takes a number of an instance of a **cross-validation generator**. These define how the data will be separated. 

The default behavior is to use the `KFold` strategy, which is equivalent to the previous code:

```python
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=10)
results = cross_validate(modelo, X, y, cv = cv)

media = results["test_score"].mean()

std = results["test_score"].std()

```

From this, to randomize the shuffling processes you online need to set the parameter shuffle to true 

```python
cv = KFold(n_splits=10, shuffle=True)
results = cross_validate(modelo, X, y, cv = cv)
```


## 🔷 Stratified K-fold Cross Validation
As we've discussed previously, randomness can be a problem if you get unlucky. Or maybe or dataset may be unbalanced for the different classes, which will cause poor and biased evaluation and training of the model.

A very common (and good) way to solve this problem is with stratification, where you guarantee that the proportion of the predicted class will be preserved through the different divisions of the dataset. 

- This method, combined with the shuffling, makes a strong cross validation procedure.

### 🟢 In Python
To use stratification in cross validation we'll use a different cross-validation generator, in this case it'll be the `StratifiedKFold`


## 🔷 Grouping in Cross Validation
Data can usually be separated into groups (car models, patients, etc.). When you use stratification and shuffling you select entries from these groups at random. 
- This means that, you're not testing if your model is capable of predicting new groups, since he has already seen all of them. This is sort of like data leaking, you're passing information that should be reserved for testing. 

To better understand your model's performance with respect to new groups (or classes) of entries, we need to do a cross validation by groups. 
- Otherwise you'll get unrealistic good results, since it'll be learning from what it should predict.

### 🟢 In Python
As expected, there's an implementation of grouping k-fold cross validation in scikit-learn:

```python
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=10)
modelo = DecisionTreeClassifier(max_depth=2)
results = cross_validate(modelo, X_azar, y_azar,
                         groups = dados.modelo, cv = cv)

imprime_resultados(results)
```

In the case above we're grouping the data my the column "dados.modelo"