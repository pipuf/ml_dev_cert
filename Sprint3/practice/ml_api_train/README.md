# ML API

This code contains the app described at the end of the "Serving ML models with APIs" lecture. It trains a model to predict the probability of someone having diabetes and stores the predictions in an SQLite database following a monolithic approach.

The model was created using Python 3.8.2

You may have to adjust the code to work with other versions of Python.

## Instructions

1. First train the model by running the following command:

```bash
python train_model.py
```

Once you finish the model and processor will be saved in the 'assets' folder.

2. Run the following command to start the API:

```bash
fastapi dev main.py
```