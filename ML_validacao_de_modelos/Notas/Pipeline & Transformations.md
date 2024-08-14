
# 🔵 Preprocessing Transformations

Its very common to apply transformation on the preprocessing phase of training a model. These transformations can be scaling, normalization, treating missing information, etc. 

The key aspect is that these transformations should be applied only on the training features not on the test. 
- Thus, for a process like cross validation you'd need to fit and transform the training data individually for each split. You can't do it on the whole dataset since it'd be a kind of data leaking.


# 🔵 Pipeline

In scikit-learn, a pipeline is a sequence of procedures to be applied. This may include, but is not limited to: transformation, functions, predictions, etc. 

Pipelines can be used as transformer and even as models. This means that can be passed to procedures like cross-validation or used by itself. 


### 🟢 In Python
For example, lets use a pipeline to apply a standard scaler and then run a cross validation procedure in a `SVC` model:

```python
from sklearn.pipeline import Pipeline

np.random.seed(SEED)
scaler = StandardScaler()
modelo = SVC()

pipeline = Pipeline(
    [("transformação", scaler), ("estimador", modelo)]
)

cv = GroupKFold()
results = cross_validate(pipeline, X_azar, y_azar, cv=cv, groups=dados.modelo)
imprime_resultados(results)
```

Notice that we've passed the pipeline in place of the model to the `cross_validation`. This means that it'll first apply the standard scaler and then execute the model as one'd expect. 