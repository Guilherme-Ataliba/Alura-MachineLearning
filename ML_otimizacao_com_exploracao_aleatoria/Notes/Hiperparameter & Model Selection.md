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


## 🔷 Overfit in Hyperparameters
> Grid Search + Cross Validation = Overfitting de hiperparâmetros. Solução: 2 cross validation

The process of hiperparameter tuning is also called model selection, since a model with a different set of hiperparameters is, by all means, a different model. 

When you try to find the best hiperparameters through a procedure like grid search, the model is also capable to fall into **overfit**, but this time cause by an **overfit in hiperparameters**. 
A procedure that uses grid search + cross validation to find the best set of hiperparameters utilizes the same data to tune the model parameters and to evaluate the model. Its training a model without splitting the data into training and testing sets.
- This process may be referred to **data leaking**, since the training data is leaking into the testing data and the model becomes prone to overfitting. 


### 🔹 Nested Cross Validation
To avoid this problem we use the technique called nested cross validation, which uses a series of train/validation/test set splits.
1. In the inner loop the score is maximized by fitting a model to each training set
2. Then, this model's hyperparameters are maximized over the validation set
3. Finally, In the outer loop, generalization error is estimated by averaging test scores over several dataset splits

In short, the inner loop trains the model and finds the best hiperparameter set, then the outer loop trains that model on different datasets to find a general error estimate.

#### 🟢 In Python

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


# 🔵 Random Search

This technique also intends to find the best set of hyperparameters, in this case using randomness. Some strong points are
- This procedure allows for the user to define the computational time / resources to spend on searching the parameters, since it allows for one to set the number of iterations, that is independent of the parameter space.
- Its still possible to make it parallel and to further optimize the procedure

In this process you also need to define a parameter space to search, except that here you define an interval to search into, instead of the specific points. 
- This also means that the space is't required to be discrete anymore, even though it can be. 

#### 🔴 Nested Cross Validation
Of course the nested cross validation is still required, since it is part of every hyperparameter tuning pipeline.


### 🟢 In Python
It works just as `GridSearchCV` does. Following there's nested cross validation with randomized search.

#### Discrete Grid
```python
from sklearn.model_selection import RandomizedSearchCV, KFold

SEED = 301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth": [3,5],
    "min_samples_split": [32, 64, 128],
    "min_samples_leaf": [32, 64, 128],
    "criterion": ["gini", "entropy"]
}

  

busca = RandomizedSearchCV(DecisionTreeClassifier(), espaco_de_parametros,
                     cv = KFold(n_splits=5, shuffle=True), random_state=SEED,
                     n_iter = 16)
busca.fit(X_azar, y_azar)
resultados = pd.DataFrame(busca.cv_results_)

scores = cross_val_score(busca, X_azar, y_azar,
                          cv = KFold(n_splits=5, shuffle=True))
imprime_scores(scores)
```

#### Continuous Grid
The following code uses `scipy`'s `randint` to generate random number as it is called. As usual, you define the number of points to search with the parameter `n_iter`.

```python
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint

SEED = 301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth": [3, 5, 10, 15, 20, 30, None],
    "min_samples_split": randint(32, 128),
    "min_samples_leaf": randint(32, 128),
    "criterion": ["gini", "entropy"]
}  

busca = RandomizedSearchCV(DecisionTreeClassifier(), espaco_de_parametros,
                     cv = KFold(n_splits=5, shuffle=True), random_state=SEED,
                     n_iter = 16)
                     
busca.fit(X_azar, y_azar)
resultados = pd.DataFrame(busca.cv_results_)
resultados.head()


scores = cross_val_score(busca, X_azar, y_azar,
                          cv = KFold(n_splits=5, shuffle=True))
imprime_scores(scores)

```


# 🔵 Model Selection Without Cross Validation
Up until now we have used **nested cross validation** in combination with search techniques to tune our model's hyperparameters. However, sometimes it isn't possible to execute cross validation (usually for computational limitations). 

In this cases we must recur to old division of data, but now we must insert the validation part, which the outer loop of nested cross validation is responsible. Then, the idea is to split the data into three subsets:
1. Training
2. Testing
3. Validation

Por exemplo, digamos que você quer 
- 0.6 Treino
- 0.2 Teste
- 0.2 Validação
Você fará:
1. Reserve 20% do total para validação e o restante para Treino e Teste
2. Para obter 20% do total de dados para teste você deve calcular quanto é 20% em cima desses 80% que restaram. Ou seja
$$
0.2/0.8=0.25
$$
	Assim, para obter os dados de teste você deve tomar 25% dos dados treino e teste e reservar o restante dos 75% para treino.
	- Isso porque 75% do treino e teste é 60% do total, da mesma forma que 25% corresponde a 20% do total.


### 🟢 In Python
We'll use the train test split twice, the first time to separate the validation set and the second one to split train and test (don't forget to recalculate the percentage for training and testing):

```python
# 0.6 Treino
# 0.2 Teste
# 0.2 Validação

SEED = 301
np.random.seed(SEED)

# Separando a validação
from sklearn.model_selection import train_test_split

X_treino_test, X_validacao, y_treino_test, y_validacao = train_test_split(X_azar, y_azar, test_size=0.2, shuffle=True, stratify=y_azar)



from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from scipy.stats import randint

# 20% dos 80% original é 25% (0.2/0.8 = 0.25)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25)

espaco_de_parametros = {
    "n_estimators": [10, 100],
    "max_depth": [3, 5, 10, 15, 20, 30, None],
    "min_samples_split": randint(32, 128),
    "min_samples_leaf": randint(32, 128),
    "criterion": ["gini", "entropy"],
    "bootstrap": [False, True]
}
tic = time.time()

busca = RandomizedSearchCV(RandomForestClassifier(), espaco_de_parametros,
                     cv = split, n_iter=20,
                     random_state=SEED)
busca.fit(X_treino_test, y_treino_test)
```

Then you'll only to actually apply the validation:

```python
scores = cross_val_score(busca, X_validacao, y_validacao,
                          cv = split)
```