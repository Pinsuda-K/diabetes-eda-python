### diabetes-eda-python
# Machine Learning Practice — Diabetes Dataset (Google Colab, Python)
![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Type](https://img.shields.io/badge/project-learning-blue)
![ML](https://img.shields.io/badge/topic-machine%20learning-lightgrey)
![Python](https://img.shields.io/badge/language-python-blue)
![Colab](https://img.shields.io/badge/notebook-colab-orange)

> This project was completed as part of a personal assignment to practice data analysis and model building using Python. The dataset contains anonymized medical information related to diabetes. The goal of the assignment was **not clinical research**, but rather to:
> ✅ apply Python for data cleaning and EDA
> ✅ build and compare basic machine learning models
> ✅ evaluate and interpret results

### Objective
- Use a diabetes dataset to practice the full machine learning workflow, from data cleaning → visualization → model building and comparison (Logistic Regression, Decision Tree, etc.).
---
### Tools & Tech:
| Tool              | Purpose                            |
|-------------------|-------------------------------------|
| `Python`            | Programming language                |
| `Pandas`            | Data cleaning/manipulation          |
| `Matplotlib` / `Seaborn` | Visualization / EDA              |
| `Scikit-Learn`      | ML models (LogReg, DecisionTree, ...) |
| `Google Colab`      | Notebook environment                |
---
### Workflow Summary
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
### Key Takeaways (from this exercise)
- **Linear Regression delivered the best performance** among the three tested models (R² = **0.45**, mean cross-validated R² ≈ **0.48 ± 0.05**).
- **Decision Tree** and **Random Forest** did not provide significant improvement over the linear model — suggesting that the relationship between input features and the target variable in this dataset is mostly linear.
- **Cross-validation** confirms that **Linear Regression** is the most stable model (lowest standard deviation of R²), while Decision Tree produced the most variability.
- The dataset contained **no missing values** and was **standardized** prior to modeling.
- bmi and s5 (a serum measurement) showed the **strongest positive correlation** with the target (0.59 and 0.57 respectively).
- Random Forest feature importance also identified bmi and s5 as the most influential predictors.
- Overall, the dataset has **moderate predictive capability** — visualization of actual vs. predicted values shows that even the best-performing model still has room for improvement.

### สรุปประเด็นสำคัญจากแบบฝึกหัด (ภาษาไทย)
- โมเดล **Linear Regression** ให้ผลลัพธ์ดีที่สุด ในบรรดา 3 โมเดลที่ทดสอบ (R² = **0.45** และค่าเฉลี่ย Cross-Validation ≈ **0.48 ± 0.05**)
- **Decision Tree** และ **Random Forest** ไม่ได้ให้ผลที่ดีขึ้นอย่างมีนัยสำคัญ ซึ่งบ่งบอกว่า ความสัมพันธ์ระหว่าง feature กับ target มีลักษณะเป็นเชิงเส้น (linear) เป็นหลัก
- จากผล **Cross-Validation** แสดงว่า **Linear Regression** เป็นโมเดลที่เสถียรที่สุด (ค่าเบี่ยงเบนมาตรฐานต่ำที่สุด) ในขณะที่ Decision Tree มีความผันผวนมากกว่า
- Dataset นี้ **ไม่มี missing value** และมีการปรับมาตรฐาน (standardize) ก่อนการ train โมเดล
- **ตัวแปรที่มีความสัมพันธ์กับ target สูงที่สุด** คือ bmi และ s5 (ค่าความสัมพันธ์ 0.59 และ 0.57 ตามลำดับ)
- การวิเคราะห์ feature importance ของ Random Forest ก็แสดงให้เห็นว่า bmi และ s5 เป็น feature ที่มีอิทธิพลสูงที่สุดเช่นกัน
- โดยรวมแล้ว dataset นี้มี **ความสามารถในการพยากรณ์ในระดับปานกลาง** – ถึงแม้ Linear Regression จะให้ผลดีที่สุด แต่การเปรียบเทียบ actual vs predicted ยังเห็นว่ามีช่องว่างบางส่วนที่สามารถปรับปรุงได้ในอนาคต
---
### Notebook
### 👉 View the full code and analysis in Google Colab: https://colab.research.google.com/drive/1-2G1mkhT9AJHWs9Kz0MCbKWa3AoTbEN6?usp=sharing
---
### Possible Improvements (next iteration)
- Hyperparameter tuning  
- Add support vector machine (SVM) model for comparison  
- Implement K-fold cross validation  
- Build simple Streamlit UI for prediction
---
