This chapter is dedicated to techniques of preprocessing the data, like removing "useless" words, making important words more frequent, etc.


# 🔵 Stop Words
In the context of Natural Language Models, words that have no meaning to a machine learning model and are basically useless in that sense, are called stop words. 
- These are the words that usually show up more in a text, since they are articles, conjunctions, etc.

Since this is a deeply studied topic, there is a list of stop words publicly available that can be used to remove them from the input beforehand. 

### 🟢 In Python
Following there's an example of a script that remove stop words from the data and creates a new pandas column with it:

```python
stop_words = nltk.corpus.stopwords.words("portuguese")

for item in resenha.text_pt:
    nova_frase, frase_processada = list(), list()
    token_espaco = nltk.tokenize.WhitespaceTokenizer()
    palavras_texto = token_espaco.tokenize(item)

    for palavra in palavras_texto:
        if palavra not in stop_words:
            nova_frase.append(palavra)

    frase_processada.append(" ".join(nova_frase))

resenha["tratamento_1"] = frase_processada
```
- Notice that to access the stop words from `nltk` you have to download it using a method from the package.