
TF-IDF is another algorithm that vectorizes words, but differently from the bag of words, this model tries to apply weights to more important words. 

# 🔵 The Process
Instead of simply counting the frequency of each word and assigning this frequency value to a table, TF-IDF normalizes the frequency value. Meaning it divides the frequency in each word by the number of times that word has appeared in the whole dataset, among all words.

This process is based on the characteristic that words that appear too much in every different entry loose their differentiation capability, their meaning.
> *"If you say something too much it looses its meaning"*

On the other hand this algorithm favors words that appear less frequently, and usually these are the words that carry most meaning.
![[Pasted image 20240817173824.png]]

### 🟢 In Python
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_tratado = tfidf.fit_transform(resenha["tratamento_5"])

X_train, X_test, y_train, y_test = train_test_split(tfidf_tratado,
                                                    resenha["classificacao"],
                                                    random_state=42)

regressao_logistica.fit(X_train, y_train)
regressao_logistica.score(X_test, y_test)
```