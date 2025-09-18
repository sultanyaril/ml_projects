# Digit Recognition

Purpose
- Train and evaluate convolutional neural networks to classify 28x28 handwritten digits.

Quick run
- Use the notebook `digit_recognition.ipynb` for exploration and training.
- To run the simple demo app (draw and predict):
  ```powershell
  python app.py
  ```

Key files
- `digit_recognition.ipynb` — training, evaluation and analysis.
- `app.py` — small drawing app to try the trained model interactively.
- `trained_model.torch`, `trained_model_black_and_white.torch` — saved model weights.
- `train.csv`, `test.csv` — example CSV datasets (if present).

Notes
- Preprocessing matters: grayscale vs black-and-white input may require retraining or conversion.
