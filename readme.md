# Taiwan Food 101
這是北科深度學習的第二次作業，準確率約為95.4%.
為了減少處理跟硬體溝通的程式碼，選擇使用了pytorch lightning framework.

# How to start
## Install the dependency packages
```pip install -r ./requirements.txt```
Note: If you are using AMD GPU you need to manaually install ROCm version pytorch by yourself

## Train
Run the main.py and modify the hyperparameters in main.py.
``` python main.py```

You also modify the model

## Inference
```python test.py```