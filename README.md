# ScriptSense – Handwritten Text Recognition (HTR)

**ScriptSense** is a deep learning-based system that reads handwritten English text from images. It uses a hybrid **CNN + BiLSTM + Transformer** architecture to deliver accurate results.


## What's Inside?

- `ScriptSense_HTR_Final.ipynb` – The complete notebook with everything: preprocessing, training, evaluation, and prediction.


## How to Run?

### Option 1: Google Colab (Easy & Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook
3. Run cells one by one – it takes care of setup, training, and predictions

> Note: Make sure to upload or link the IAM dataset (explained in the notebook).

### Option 2: Run Locally
1. Install Python 3.8+
2. (Optional) Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebook using Jupyter and run it.


## Requirements

Main libraries:
- `tensorflow`
- `numpy`
- `opencv-python`
- `matplotlib`, `seaborn`
- `scikit-learn`

All necessary installs are handled in the notebook if you're on Colab.


## Dataset

We use the **IAM Handwriting Database**.  
We imported the same from [Kaggle](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database) If you require the original dataset, you’ll need to download it from [here](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and set the correct path.


## Metrics

Evaluation includes:
- Exact Match Accuracy  
- F1 Score  
- AUC Curve  
- Confusion Matrix  


## Authors

This project was developed by a team of undergraduate students from Dayananda Sagar University:
- Ansuman Panda
- Mohamed Aasim Amjad
- Pranav Vinod Pillai
- R S Chiraag


## License

MIT – free to use, modify, and share.
