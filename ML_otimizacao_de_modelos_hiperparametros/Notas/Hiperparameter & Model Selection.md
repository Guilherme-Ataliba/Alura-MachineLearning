# 🔵 Correlation & Visualization ($nD$)
For hiperparameter tuning in more than one dimension its very common to look at a **correlation matrix**. This matrix doesn't show causality (meaning that one factor is the cause of the other), but instead it shows that for that particular case the data seems to be correlated.


### 🟢 In Python
This is a statistical test, not a ML model, consequently, it is easily accessible through pandas `.coor()` in a dataframe.

Also, a very common manner to visualize this data is with a heatmap (such as the one implemented by seaborn).

```python
corr = resultados.corr()
sns.heatmap(data=corr)
```
![[Pasted image 20240813201616.png]]
- PS: Seaborn has a code that customizes the heatmap to better represent a correlation matrix. This is that code:

```python
from string import ascii_letters

mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```
- It is still a heatmap, but it has significant changes in appearance to help visualize the important information in a correlation matrix. 

Ou até um `pairplot` do seaborn, que mostra diferentes gráficos entre todas as features no conjunto de dados, permitindo diversas visualizações.

```python
sns.pairplot(resultados)
```
![[Pasted image 20240813201801.png]]


# 🔵 Hiperparameter Search - Grid Search

Grid search is a technique to explore hiperparameter space in search for the best combination of them. As you'd expect, you must define possible values for each feature, that in combination will create a grid for search.

### 🟢 In Python
Something else that's interesting is that GridSearchCV also allows for cross validation in the processes of hiperparameter search (which is pretty required to avoid randomness) and also groups of data.

```python
from sklearn.model_selection import GridSearchCV

SEED = 301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth": [3,5],
    "min_samples_split": [32, 64, 128],
    "min_samples_leaf": [32, 64, 128],
    "criterion": ["gini", "entropy"]
}

busca = GridSearchCV(DecisionTreeClassifier(), espaco_de_parametros,
                     cv = GroupKFold(n_splits=10))
busca.fit(X_azar, y_azar, groups=dados.modelo)
resultados = pd.DataFrame(busca.cv_results_)
```


# 🔵 Overfit in Hyperparameters
> Grid Search + Cross Validation = Overfitting de hiperparâmetros. Solução: 2 cross validation

The process of hiperparameter tuning is also called model selection, since a model with a different set of hiperparameters is, by all means, a different model. 

When you try to find the best hiperparameters through a procedure like grid search, the model is also capable to fall into **overfit**, but this time cause by an **overfit in hiperparameters**. 
A procedure that uses grid search + cross validation to find the best set of hiperparameters utilizes the same data to tune the model parameters and to evaluate the model. Its training a model without splitting the data into training and testing sets.
- This process may be referred to **data leaking**, since the training data is leaking into the testing data and the model becomes prone to overfitting. 


## 🔷 Nested Cross Validation
To avoid this problem we use the technique called nested cross validation, which uses a series of train/validation/test set splits.
1. In the inner loop the score is maximized by fitting a model to each training set
2. Then, this model's hyperparameters are maximized over the validation set
3. Finally, In the outer loop, generalization error is estimated by averaging test scores over several dataset splits

In short, the inner loop trains the model and finds the best hiperparameter set, then the outer loop trains that model on different datasets to find a general error estimate.

### 🟢 In Python

```python
from sklearn.model_selection import GridSearchCV, KFold

SEED = 301
np.random.seed(SEED)


# Inner Loop
espaco_de_parametros = {
    "max_depth": [3,5],
    "min_samples_split": [32, 64, 128],
    "min_samples_leaf": [32, 64, 128],
    "criterion": ["gini", "entropy"]
}  

busca = GridSearchCV(DecisionTreeClassifier(), espaco_de_parametros,
                     cv = KFold(n_splits=10, shuffle=True))
busca.fit(X_azar, y_azar)
resultados = pd.DataFrame(busca.cv_results_)

# End of Inner loop
# Out Loop

from sklearn.model_selection import cross_val_score
scores = cross_val_score(busca, X_azar, y_azar,
                          cv = KFold(n_splits=5, shuffle=True))


```