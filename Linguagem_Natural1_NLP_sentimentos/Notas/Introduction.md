Problems like text generation or emotion analysis have as study object texts. This means that the input variable is a text (usually big) and the output could be a class or another text. 

The problem is that every ML model studied so far is entirely mathematical, and is supposed to be applied on numbers, vectors, tensor, etc. 
> So the purpose of **natural language models** is to transform text to mathematical entities in a way that the meaning of the message is preserved.

- Later one may apply any ML model or NN to the transformed data, since it is already in a form that the algorithms take. The actual model to choose will depend on particularities of the problem, as usual. 

### 🔹 Standard Considerations

#### 1. Understand your data
Depending on the problem the text structure can vary greatly, as well as the purpose of the analysis. This said, read some of your data and try to gather information of how it is written and structure. 

The ideia is this phase is to gather information that could be used as *hints* or guidelines on constructing the ML model.


# 🔵 Natural Language Processing
NLP is an area of artificial intelligence that tries to establish a connection between human language and computer language. 
- Or more commonly, how to make computers understand human language.

## 🔷 Bag of Words
This is the simplest form to represent words/phrases as mathematical entities, in this case, vectors. 

The ideia is to create a vocabulary table (bag of words). Every entry in the input data will be passed through the table, that counts the number of times each word is present in that text.
![[Pasted image 20240816194034.png]]
- Notice that this representation is very powerful in a way that it expresses words as input that can be read by ML models and also expresses the number of times each word appears.
	- This makes possible to assign meaning to each entry in the vector and then some kind of weight that relates to the number of times that word appears. 
- Since this matrix would end up with loads of zeros (that doesn't have much meaning) it's very common (and recommended) to use sparse matrices.

**Training**: Its important to notice that during the training process you'll build your bag of words. This implies that if the training data is not embracing enough it could let some important words outside of the model. As example the "péssimo" above.

### 🔹 Roadblocks 
Given that text databases usually have loads of different words, most of the time you won't be able to construct a vocabulary with all of them (and maybe that's not even useful). 
- A usual way to deal with this problem is to select a number $N$ of the most frequent words to construct the vocabulary. 


So the question arises of how to choose the most important and representative words in the dataset. After all, its not unreasonable to suppose that "meaningless words" like articles and connections (a, with, like, the, i, its) will show up more frequently than words that actually have meaning.

- Besides that, the actual definition of important words will vary depending on the problem you're trying to solve. In some situations some words will be more significant than others.



