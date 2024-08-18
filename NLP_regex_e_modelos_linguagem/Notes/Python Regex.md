This chapter will discuss the specifics of regex in python, not how regex work.

# 🔵 `re` - Important Features
This is the python library to use regex. Most methods and features are very straight forward, but here we'll discuss some important aspects of the library.

## 🔷 Compiling
Every time you call a regex method and pass a regex string for it, `re` will have to first compile the regex and then use it to do the task you've asked for.

The problem is, if you must do the same task multiple times, as in looping a database, this can become computationally expensive. The solution is then to compile the regex beforehand and then use it to call the methods you need. 
- The time differences can get pretty large for big datasets.

### 🟢 In Python
```python
import re

regex = re.compile(r"70")
regex.search("684746216489798770")
```