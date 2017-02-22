setup (first time after clone)
```bash
virtualenv -p /usr/local/bin/python3.6 venv
. venv/bin/activate
pip install -r requirements.txt
```

naive (naive_mnist_softmax.py):
```python
step 0, training accuracy: 0.12
step 100, training accuracy: 0.95
step 200, training accuracy: 0.93
step 300, training accuracy: 0.95
step 400, training accuracy: 0.88
step 500, training accuracy: 0.92
step 600, training accuracy: 0.92
step 700, training accuracy: 0.84
step 800, training accuracy: 0.89
step 900, training accuracy: 0.94
step 1000, training accuracy: 0.93
step 1100, training accuracy: 0.91
step 1200, training accuracy: 0.93
step 1300, training accuracy: 0.92
step 1400, training accuracy: 0.95
step 1500, training accuracy: 0.98
step 1600, training accuracy: 0.95
step 1700, training accuracy: 0.9
step 1800, training accuracy: 0.93
step 1900, training accuracy: 0.9
test accuracy: 0.9183
```

2 layer convolutional (deep_mnist_softmax.py):
```python
step 0, training accuracy: 0.04
step 100, training accuracy: 0.84
step 200, training accuracy: 0.94
step 300, training accuracy: 0.88
step 400, training accuracy: 0.98
step 500, training accuracy: 0.94
step 600, training accuracy: 0.96
step 700, training accuracy: 0.96
step 800, training accuracy: 0.92
step 900, training accuracy: 0.98
step 1000, training accuracy: 0.98
step 1100, training accuracy: 0.98
step 1200, training accuracy: 1
step 1300, training accuracy: 0.94
step 1400, training accuracy: 1
step 1500, training accuracy: 0.96
step 1600, training accuracy: 1
step 1700, training accuracy: 0.98
step 1800, training accuracy: 0.98
step 1900, training accuracy: 1
test accuracy: 0.9766
```

[paper using similar approach to AD vs NC f/sMRI](http://biorxiv.org/content/biorxiv/early/2016/08/30/070441.full.pdf)
