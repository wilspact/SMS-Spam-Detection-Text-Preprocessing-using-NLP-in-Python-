# 📬 SMS Spam Detection - Text Preprocessing using NLP in Python

This project focuses on the **text preprocessing pipeline** for SMS spam detection using Python and Natural Language Processing (NLP). The goal is to clean and transform raw SMS messages into a format suitable for training machine learning models for classification tasks (spam vs ham).

---

## 🛠️ Technologies Used

- Python
- Pandas
- NLTK (Natural Language Toolkit)
- Regular Expressions (re)
- Spyder IDE (Anaconda)
- CSV Dataset

---

## 📂 Dataset

The dataset used is `spam.csv`, which contains SMS messages labeled as either:
- `spam` – unwanted or promotional content
- `ham` – legitimate messages

---

## 📊 Preprocessing Pipeline

### ✅ 1. Data Loading
Load the dataset using `pandas` and select only the necessary columns: `Category` and `Message`.

### ✅ 2. Remove Punctuation
Use `string.punctuation` to filter out special characters from the text.

### ✅ 3. Convert to Lowercase
Convert all text to lowercase to standardize it.

### ✅ 4. Tokenization
Split the sentence into words (tokens) using `re.split()`.

### ✅ 5. Remove Stopwords
Use NLTK's English stopword list to remove common but meaningless words like `the`, `is`, `and`, etc.

### ✅ 6. Stemming
Apply **SnowballStemmer** to reduce words to their base or root form (e.g., `playing` → `play`).

### ✅ 7. Lemmatization
Use **WordNetLemmatizer** to further refine root forms based on dictionary meanings (e.g., `better` → `good`).

---

## 🖼️ Sample Output (Shown in Spyder IDE)

The processed output includes:
- Original message
- Cleaned text
- Tokenized words
- Without stopwords
- Stemmed words
- Lemmatized words



---

## 📌 Future Work

- Add **TF-IDF or CountVectorizer** for feature extraction.
- Train spam detection models (e.g., Naive Bayes, SVM).
- Build a web or mobile interface to test real-time messages.

---

## 💡 Learnings

This project gave hands-on experience in:
- Cleaning raw text data
- Understanding common NLP preprocessing steps
- Using NLTK for stopwords, stemming, and lemmatization
- Preparing data for machine learning classification

---

## 📁 Folder Structure

