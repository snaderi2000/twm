# twm

The goal of this project is to reimplement this [paper](https://arxiv.org/pdf/2502.01591).

We will first try and recreate their model free method before adding the model to generate augmented data for the training.

`conda create -n twm_env python=3.10`

Install requirnment `pip install -r requirements.txt`

then 

```pip install -e .```

## Tests 
```pytest -v tests/models/test_impala_cnn.py```

