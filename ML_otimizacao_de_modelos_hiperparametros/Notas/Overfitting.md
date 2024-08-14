Something important to remember about overfitting is that the performance decreases for the test data. But this only happens because it has learning "too well" the training data. Thus, as the model gets more and more overfitted we expect the metrics on the test data to be come worse while the metrics on the training data to become better and better (with overfit).

# 🔵 Visualizing Overfitting (1D)
A somewhat interesting way to visualize overfitting (with respect to a hyperparameter) is to vary a hyperparameter and save a metric score for each run, then you plot it against hyperparameter. 

### 🟢 In Python
![[Pasted image 20240813192104.png]]


