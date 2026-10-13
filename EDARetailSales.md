# Sentiment Analysis Using Machine Learning

## Project Overview
This project uses Natural Language Processing (NLP) and machine learning to classify text into sentiment categories: **positive, neutral, and negative**. The notebook explores the dataset, checks missing values, analyzes sentiment distribution, converts text into numerical features using TF-IDF, and trains a Logistic Regression model to predict sentiment.

## Objectives
- Explore and understand the sentiment dataset.
- Check the dataset shape and missing values.
- Remove rows where the text field is missing.
- Analyze sentiment distribution and basic text statistics.
- Convert text into numerical features using TF-IDF.
- Train and evaluate a Logistic Regression classifier.
- Visualize sentiment counts, model predictions, and sentiment patterns across tweet time and user age groups.

## Tools and Technologies
- **Python**
- **Pandas** and **NumPy** — data handling and numerical operations
- **Matplotlib** and **Seaborn** — visualizations
- **Scikit-learn** — text vectorization, model training, and evaluation
- **Jupyter Notebook** — interactive development environment

## Dataset
The notebook expects a CSV file named `sentimental_analysis_dataset.csv.csv` with columns used in the analysis, including:
- `text` — the text to analyze
- `sentiment` — the sentiment label (`positive`, `neutral`, or `negative`)
- `Time of Tweet` — time-of-day category
- `Age of User` — user age-group category

Place the CSV file in your project folder. Update the `file_path` variable in the notebook to match the file's location. Do not upload a dataset unless you have permission to share it.

## Workflow
1. **Load the data** using Pandas.
2. **Inspect data quality** by checking the dataset shape and missing values.
3. **Clean the data** by removing rows with missing text.
4. **Explore the data** by counting sentiment labels and calculating average text length and word count.
5. **Split the dataset** into training and testing sets using an 80/20 split, with stratification to preserve class proportions.
6. **Convert text to features** using `TfidfVectorizer`, limited to 5,000 features and with English stop words removed.
7. **Train the model** using Scikit-learn's Logistic Regression classifier.
8. **Evaluate performance** using accuracy, a classification report, and a confusion matrix.
9. **Visualize patterns** using plots for sentiment counts, time of tweet, and user age group.

## How to Run
1. Download or clone this repository.
2. Install Python 3.9 or newer (a recent Python version is recommended).
3. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
4. Place the dataset CSV file in the project folder.
5. Open a terminal in that folder and run:
   ```bash
   jupyter notebook
   ```
6. Open `OIBSIPDataAnalytics-L1-sentimentalanalysis.ipynb`.
7. Update the CSV path in the data-loading cell, then run all notebook cells from top to bottom.

## Evaluation
The notebook reports:
- **Accuracy** — overall proportion of correct predictions.
- **Precision, recall, and F1-score** — class-wise classification performance.
- **Confusion matrix** — compares true labels with predicted labels.

The actual performance values depend on the dataset and the run; use the notebook's output rather than assuming a fixed accuracy.

## Project Structure
```text
.
├── OIBSIPDataAnalytics-L1-sentimentalanalysis.ipynb
├── sentimental_analysis_dataset.csv.csv
└── README.md
```

## Possible Improvements
- Add more text preprocessing, such as handling URLs, mentions, punctuation, and repeated characters.
- Compare Logistic Regression with other classifiers such as Multinomial Naive Bayes or Linear SVM.
- Save the trained model and TF-IDF vectorizer for reuse.
- Test the model on new, unseen text.

## Author
**Anamika Binu**

## Acknowledgement
This project was completed as a sentiment analysis task under the OIBSIP Data Analytics project series.
