# 🔵 Ngrams
Instead of saving each word in a text separately, ngrams create a structure that tries to preserve the memory from the text. This structure consists of a word and $n$ words that came before it, thus creating the "memory".
- Bigrams are commonly used, they are composed of two words.

Its important to notice that the size of the input features will be multiplied by $n$, implying that this technique can be very computationally expensive.

#### Example
Tokenized phrase
$$
\text{['Assisti', 'um', 'ótimo', 'filme', '.']}
$$
Bigram 
$$\text{[('Assisti', 'um'), ('um', 'ótimo'), ('ótimo', 'filme'), ('filme', '.')]}$$

### 🟢 In Python
Its important to consider that ngrams usually don't appear as much as single words, for that reason you should consider increasing the max number of features allowed in your model if you want to observe any change.

Bellow we use the method `tfidf` instead of the bag of words, for more information look [[TF-IDF]]. Usually these methods have a built-in ngrams option.

```python
tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))

vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
  
X_train, X_test, y_train, y_test = train_test_split(vetor_tfidf,
                                                    resenha["classificacao"],
                                                    random_state=42)

  
regressao_logistica.fit(X_train, y_train)

regressao_logistica.score(X_test, y_test)
```

## 🔷 Fake Char
A very common problem when using ngrams is that characters that appear at the end of a sentence will show up less than characters in the middle, given how the ngrams are created.
$$\text{[('Assisti', 'um'), ('um', 'ótimo'), ('ótimo', 'filme'), ('filme', '.')]}$$
- In the example above, every word except "Assisti" and "." appears twice (bigram)

This process you make the frequency of these characters artificially lower and thus make the model less likely to predict them.

To solve this problem we must inform to the model (include in the database) where is the start and end of a sentence. And to achieve this we use false characters at the end and start of sentences, that actually have no meaning to the model.

### 🟢 In Python
The following example uses false characters to construct a bigram (n=2)

```python
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends

texto_teste = "alura"

list(bigrams(pad_both_ends(texto_teste, n=2)))
```