This chapter is dedicated to techniques of preprocessing the data, like removing "useless" words, making important words more frequent, etc.


# 🔵 Stop Words
In the context of Natural Language Models, words that have no meaning to a machine learning model and are basically useless in that sense, are called stop words. 
- These are the words that usually show up more in a text, since they are articles, conjunctions, etc.

Since this is a deeply studied topic, there is a list of stop words publicly available that can be used to remove them from the input beforehand. 


## 🔷 Punctuation
For the same reason we've removed the stop words, we also must remove punctuation. Furthermore, they not only occupy positions that could be more relevant for another words, they "create" another words, since
$$\text{movie.} \neq \text{movie,}$$
## 🔷 Accents
Mistyping some words in languages that have accent could generate problems in detection of some words. For that reason, and to removed repetitive meaningless words, we should remove punctuation from every word in the vocabulary.  

## 🔷 Lowercase
Since some methods of removing stopwords and other elements are only describe for lowercase letter, and also because uppercase and lowercase usually don't make a significant difference in most contexts, it is useful to make the whole dataset lowercase.


## 🔷 Stemming
Stemming is a text processing technique that aims to reduze words to its "root", called stem. This is the process of removing variations from words that have the same meaning, like conjugations, plural, verbal times, etc. The idea is to have only a single "stem" that represents every other similar word.
![[Pasted image 20240817172444.png]]
Its important to remember that there are a few different stemming algorithms, so choosing one is very important. Besides that, they are different from every language, so if you're working if a different language you must find another stemming algorithm.

### 🟢 In Python
Following there's an example of a script that remove stop words from the data and creates a new pandas column with it. This script also removes:
- Punctuation
- Accents
- Lowercase

```python
from string import punctuation
import unidecode

stop_words = nltk.corpus.stopwords.words("portuguese")

pontuacao = [ponto for ponto in punctuation]
pontuacao_stopwords = stop_words + pontuacao

stop_words_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

resenha["tratamento_3"] = [unidecode.unidecode(texto) for texto in resenha["tratamento_2"]]

stemmer = nltk.RSLPStemmer()

token_espaco = nltk.tokenize.WordPunctTokenizer()
for item in resenha["tratamento_3"]:
    nova_frase, frase_processada = list(), list()
    item = item.lower()
    palavras_texto = token_espaco.tokenize(item)

    for palavra in palavras_texto:
        if palavra not in stop_words_sem_acento:
			stem = stemmer.stem(palavra)
            nova_frase.append(stem)

    frase_processada.append(" ".join(nova_frase))

resenha["tratamento_3"] = frase_processada
```
- Notice that to access the stop words from `nltk` you have to download it using a method from the package.


# 🔴 Preprocessing Checklist
Combined with the previous python code, here goes a checklist for data manipulation and preprocessing for natural language models, using **regex**:

```python
regex_HTML = re.compile(r"<.*?>")
regex_code = re.compile(r"<code>(.|\n)*?</code>")
regex_pontuacao = re.compile(r"[^[\w\s]]")
regex_digitos = re.compile(r"\d+")
regex_espaco = re.compile(r" +")
regex_quebra_linha = re.compile(r"(\n)")

  

def substituir(textos, regex, subs=""):
    if type(textos) == str:
        return regex.sub(subs, textos)
    else:
        return [regex.sub(subs, texto) for texto in textos]

def minusculo(textos):
    if type(textos) == str:
        return textos.lower()
    else:
        return [texto.lower() for texto in textos]

def add_preprocessed_column(data):
    questoes_sem_code = substituir(data["Questão"], regex_code, "CODE")
    questoes_sem_code_tag = substituir(questoes_sem_code, regex_HTML)
    data["sem_code_tag"] = questoes_sem_code_tag

  
    questoes_sem_pont = substituir(questoes_sem_code_tag, regex_pontuacao)
    questoes_sem_pont_min = minusculo(questoes_sem_pont)
    questoes_sem_pont_min_dig = substituir(questoes_sem_pont_min, regex_digitos)
    questoes_sem_quebra_linha = substituir(questoes_sem_pont_min_dig,
                                           regex_quebra_linha, " ")
    questoes_sem_espaco_duplicado = substituir(questoes_sem_quebra_linha,
                                               regex_espaco, " ")

    data["dados_tratados"] = questoes_sem_espaco_duplicado
    return data
```