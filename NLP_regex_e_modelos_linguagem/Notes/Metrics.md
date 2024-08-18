Here we'll list some metrics that are specific to natural language models.

# 🔵 Perplexity
Perplexity is a performance metric for language models. it is defined as the exponential of the average negative log-likelihood of a sequence of words. If the model assigns the probability P to a sequence of N words, the perplexity PP is
$$PP = e^{-1/N\sum_i=1^N \log P(\omega_i|\omega_1, \omega_2, ..., \omega_{i-1})}$$

Perplexity can be understood as a measure of uncertainty in the models prediction. Lower perplexity indicates that the model is better at predicting the next word in a sequence
- This means that it assigns higher probabilities to words that appear in the sequence.
- Perplexity can range from 1 to infinity, where 1 indicates perfect perplexity. 

In natural language models, perplexity is particularly useful in training and hyperparameter tuning, as it provides a direct metric of how well the model should generalize to unseen data.
- And it can also be used to compare different models
- **Perplexity is the accuracy of natural language models.**

#### Infinite Perplexity
If the model has never seen the testing data that you're trying to calculate the perplexity of it will return `inf`, since the perplexity is proportional to
$$PP \approx \frac{1}{P}$$
And since the model has never seen the input, the probability is zero.



# 🔵 Laplace's Model
Laplace's model address the problem of infinite perplexity by limiting the probability in such a way that it can never be equal to zero, thus $PP \neq \infty$.