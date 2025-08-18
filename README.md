# Machine Learning Practice — Diabetes Dataset (Google Colab, Python)

This project was completed as part of a personal assignment to practice data analysis and model building using Python.  
The dataset contains anonymized medical information related to diabetes.  
The goal of the assignment was **not clinical research**, but rather to:

✅ apply Python for data cleaning and EDA  
✅ build and compare basic machine learning models  
✅ evaluate and interpret results

---

## Objective

Use a diabetes dataset to practice the full machine learning workflow, from data cleaning → visualization → model building and comparison (Logistic Regression, Decision Tree, etc.).

---

## Tools & Tech

| Tool              | Purpose                            |
|-------------------|-------------------------------------|
| Python            | Programming language                |
| Pandas            | Data cleaning/manipulation          |
| Matplotlib / Seaborn | Visualization / EDA              |
| Scikit-Learn      | ML models (LogReg, DecisionTree, ...) |
| Google Colab      | Notebook environment                |

---

## Workflow Summary

1. **Pre-processing**
   - Load dataset
   - Check and handle missing values
   - Convert data types / scale features (if needed)

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics
   - Heatmap of feature correlations
   - Distribution plots

3. **Model Training & Evaluation**
   - Train/test split  
   - Logistic Regression model  
   - Decision Tree model  
   - Compared models using accuracy + confusion matrix

4. **Conclusion**
   - Identified which variables have strong impact on prediction
   - Compared model performance
   - Reflected on improvements / next steps

---

## Key Takeaways (from this exercise)

| Aspect | Result |
|-------|-------------------------------------------|
| Best performing model | Logistic Regression (xx% accuracy)* |
| Most influential variables | Glucose, BMI, Age |
| Learning Outcome | Able to compare models and explain results |

\* you can update the value based on your result

---

## Notebook

👉 View the full code and analysis in Google Colab:  
https://colab.research.google.com/drive/1-2G1mkhT9AJHWs9Kz0MCbKWa3AoTbEN6?usp=sharing

---

## Possible Improvements (next iteration)

- Hyperparameter tuning  
- Add support vector machine (SVM) model for comparison  
- Implement K-fold cross validation  
- Build simple Streamlit UI for prediction

---

