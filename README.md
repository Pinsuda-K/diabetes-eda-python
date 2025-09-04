#### diabetes-eda-python
# Machine Learning Practice — Diabetes Dataset (Google Colab, Python)
![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Type](https://img.shields.io/badge/project-learning-blue)
![ML](https://img.shields.io/badge/topic-machine%20learning-lightgrey)
![Python](https://img.shields.io/badge/language-python-blue)
![Colab](https://img.shields.io/badge/notebook-colab-orange)

> This project was completed as part of a personal assignment to practice data analysis and model building using Python.  
> The dataset is the **Diabetes dataset from `sklearn.datasets`** containing anonymized medical features and a continuous target related to disease progression.  
> The goal of the assignment was **not clinical research**, but rather to:  
> ✅ apply Python for data exploration and EDA  
> ✅ build and compare basic regression machine learning models  
> ✅ evaluate and interpret results  

> For educational use only — do not redistribute commercially

---

### Objective
- Use the **sklearn diabetes dataset** to practice the full machine learning workflow, from data exploration → visualization → regression model building and comparison (**Linear Regression, Decision Tree Regressor, Random Forest Regressor**).

---

### Tools & Tech:
| Tool              | Purpose                            |
|-------------------|-------------------------------------|
| `Python`            | Programming language                |
| `Pandas`            | Data cleaning/manipulation          |
| `Matplotlib` / `Seaborn` | Visualization / EDA              |
| `Scikit-Learn`      | ML models (Linear Regression, Decision Tree Regressor, Random Forest Regressor) |
| `Google Colab`      | Notebook environment                |

---

### Workflow Summary
1. **Pre-processing**
   - Load dataset
   - Confirm no missing values
   - Check feature types and scaling
2. **Exploratory Data Analysis (EDA)**
   - Summary statistics
   - Heatmap of feature correlations
   - Distribution plots
3. **Model Training & Evaluation**
   - Train/test split  
   - Linear Regression model  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - Compared models using **R² score and Mean Squared Error (MSE)**
4. **Conclusion**
   - Identified which variables have strong impact on prediction
   - Compared model performance
   - Reflected on improvements / next steps

---

### Key Takeaways (from this exercise)
- **Linear Regression delivered the best performance** among the three tested models (R² = **0.45**, mean cross-validated R² ≈ **0.48 ± 0.05**).  
- **Decision Tree Regressor** and **Random Forest Regressor** did not provide significant improvement over the linear model — suggesting that the relationship between input features and the target variable in this dataset is mostly linear.  
- **Cross-validation** confirms that **Linear Regression** is the most stable model (lowest standard deviation of R²), while Decision Tree showed the most variability.  
- The dataset contained **no missing values** and was **standardized** prior to modeling.  
- **bmi** and **s5 (a serum measurement)** showed the **strongest positive correlation** with the target (0.59 and 0.57 respectively).  
- Random Forest feature importance also identified **bmi** and **s5** as the most influential predictors.  
- Overall, the dataset has **moderate predictive capability** — visualization of actual vs. predicted values shows that even the best-performing model still has room for improvement.

---

### สรุปประเด็นสำคัญจากแบบฝึกหัด (ภาษาไทย)
- โมเดล **Linear Regression** ให้ผลลัพธ์ดีที่สุด ในบรรดา 3 โมเดลที่ถูกจำลอง (R² = **0.45** และค่าเฉลี่ย Cross-Validation ≈ **0.48 ± 0.05**)  
- **Decision Tree Regressor** และ **Random Forest Regressor** ไม่ได้ให้ผลที่ดีขึ้นอย่างมีนัยสำคัญ ซึ่งบ่งชี้ว่าความสัมพันธ์ระหว่าง feature กับ target มีลักษณะเชิงเส้น (linear) เป็นหลัก  
- จากผล **Cross-Validation** แสดงว่า **Linear Regression** เป็นโมเดลที่เสถียรที่สุด (ค่าเบี่ยงเบนมาตรฐานต่ำที่สุด) ในขณะที่ Decision Tree มีความผันผวนมากกว่า  
- Dataset นี้ **ไม่มี missing value** และมีการปรับมาตรฐาน (standardize) ก่อนการ train โมเดล  
- **ตัวแปรที่มีความสัมพันธ์กับ target สูงที่สุด** คือ **bmi (ดัชนีมวลกาย)** และ **s5 (ระดับ serum ที่ 5)** (ค่าความสัมพันธ์ 0.59 และ 0.57 ตามลำดับ)  
- การวิเคราะห์ feature importance ของ Random Forest ก็แสดงให้เห็นว่า **bmi** และ **s5** เป็น feature ที่มีอิทธิพลสูงที่สุดเช่นกัน  
- โดยรวมแล้ว dataset นี้มี **ความสามารถการพยากรณ์ในระดับปานกลาง** – ถึงแม้ Linear Regression จะให้ผลดีที่สุด แต่การเปรียบเทียบ actual vs predicted ยังเห็นว่ายังคงมีช่องว่างบางส่วนที่สามารถนำไปปรับปรุงได้ต่อในอนาคต  

---

### Notebook
#### View the full code and analysis in Google Colab 👉 [Open Notebook](https://colab.research.google.com/drive/1-2G1mkhT9AJHWs9Kz0MCbKWa3AoTbEN6?usp=sharing)

---

### Possible Improvements (next iteration)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Add **Support Vector Regressor (SVR)** for comparison  
- Implement **K-Fold cross validation** (instead of simple train-test split)  
- Build a simple **Streamlit app** to input features and predict progression score  

---
