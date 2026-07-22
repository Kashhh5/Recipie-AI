# 🍽️ RECIPEAI – Smart Recipe Suggestion System

A content-based recipe recommendation system that suggests relevant recipes based on user-provided ingredients using **TF-IDF (Term Frequency–Inverse Document Frequency)** and **cosine similarity**. The application provides an intuitive interface built with **Streamlit** for searching and discovering recipes in real time.

---

## 📌 Project Overview

RECIPEAI helps users find suitable recipes by matching the ingredients they have with recipes from a dataset. It leverages Natural Language Processing (NLP) techniques to recommend the most relevant recipes quickly and efficiently.

---

## 🚀 Features

- Ingredient-based recipe recommendations
- TF-IDF text vectorization for ingredient matching
- Cosine similarity for ranking recipes
- Interactive Streamlit web application
- Fast and scalable recommendation system
- User-friendly search interface

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit

---

## 📂 Project Structure

```
RecipeAI/
│
├── app.py
├── recipes.csv
├── requirements.txt
├── README.md
│
├── model/
│   └── recommendation.py
│
└── screenshots/
    ├── Home.png
    ├── Recommendation.png
    └── Results.png
```

---

## ⚙️ How It Works

1. User enters one or more ingredients.
2. The dataset is preprocessed and cleaned.
3. TF-IDF converts recipe ingredients into numerical feature vectors.
4. Cosine similarity compares the user's ingredients with all recipes.
5. The system recommends the most relevant recipes ranked by similarity.

---

## 📊 Dataset

- Recipe dataset containing recipe names, ingredients, and related information.
- Data preprocessing included cleaning missing values, text normalization, and feature extraction.

---

## 📈 Key Highlights

- Built a content-based recommendation engine using TF-IDF.
- Implemented efficient ingredient matching using cosine similarity.
- Developed an interactive Streamlit application for real-time recommendations.
- Processed and analyzed recipe datasets for improved recommendation quality.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/RecipeAI.git
```

Navigate to the project directory:

```bash
cd RecipeAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Application Preview

### Home Page

![Home](screenshots/Home.png)

### Recipe Recommendations

![Recommendation](screenshots/Recommendation.png)

### Results

![Results](screenshots/Results.png)

---

## 🎯 Future Enhancements

- Nutritional information integration
- Cuisine-based filtering
- Recipe bookmarking
- User authentication
- AI-powered personalized recommendations
- Voice-based ingredient search

---

## 👨‍💻 Author

Kashish Bodhwani


