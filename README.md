# From Raw to Ready — Adult Census Income  

## 📌 Overview  
This project audits, cleans, and transforms the **UCI Adult Census Income dataset** into an analysis-ready format.  
The aim is to uncover socio-economic patterns affecting income (`<=50K` vs `>50K`) and provide:  
- A clean dataset  
- An EDA report with insights  
- A reproducible processing pipeline  

---

## 📊 Dataset  
- **Source Data:** [Adult Census Income (UCI ML Repository)](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)  
- **Schema:** [Column Descriptions](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)  

---

## 🛠️ Week 1: Data Loading & Basic Inspection  

We begin by loading the dataset, assigning column names, and running some initial inspection.  

### 🔹 Code
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset URL
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Column names from UCI Adult dataset
COLUMNS = [
    "age","workclass","fnlwgt","education","maritalstatus","occupation",
    "education_num","relationship","race","gender","capital_gain",
    "capital_loss","country","income","hours_per_week"
]

# Load dataset
df = pd.read_csv(
    DATA_URL,
    header=None,
    names=COLUMNS,
    na_values="?",          # Treat '?' as missing values
    skipinitialspace=True   # Remove leading spaces
)

# Preview first 5 rows
print(df.head())

# Dataset shape
print(df.shape)

# Data info
print(df.info())

# Summary statistics
print(df.describe())
 age         workclass  fnlwgt  education  maritalstatus  \
0   39         State-gov   77516  Bachelors             13   
1   50  Self-emp-not-inc   83311  Bachelors             13   
2   38           Private  215646    HS-grad              9   
3   53           Private  234721       11th              7   
4   28           Private  338409  Bachelors             13   

           occupation      education_num   relationship   race  gender  \
0       Never-married       Adm-clerical  Not-in-family  White    Male   
1  Married-civ-spouse    Exec-managerial        Husband  White    Male   
2            Divorced  Handlers-cleaners  Not-in-family  White    Male   
3  Married-civ-spouse  Handlers-cleaners        Husband  Black    Male   
4  Married-civ-spouse     Prof-specialty           Wife  Black  Female   

   capital_gain  capital_loss  country         income hours_per_week  
0          2174             0       40  United-States          <=50K  
1             0             0       13  United-States          <=50K  
2             0             0       40  United-States          <=50K  
3             0             0       40  United-States          <=50K  
4             0             0       40           Cuba          <=50K  
(32561, 15)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             32561 non-null  int64 
 1   workclass       30725 non-null  object
 2   fnlwgt          32561 non-null  int64 
 3   education       32561 non-null  object
 4   maritalstatus   32561 non-null  int64 
 5   occupation      32561 non-null  object
 6   education_num   30718 non-null  object
 7   relationship    32561 non-null  object
 8   race            32561 non-null  object
 9   gender          32561 non-null  object
 10  capital_gain    32561 non-null  int64 
 11  capital_loss    32561 non-null  int64 
 12  country         32561 non-null  int64 
 13  income          31978 non-null  object
 14  hours_per_week  32561 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
None
                age        fnlwgt  maritalstatus  capital_gain  capital_loss  \
count  32561.000000  3.256100e+04   32561.000000  32561.000000  32561.000000   
mean      38.581647  1.897784e+05      10.080679   1077.648844     87.303830   
std       13.640433  1.055500e+05       2.572720   7385.292085    402.960219   
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000   
25%       28.000000  1.178270e+05       9.000000      0.000000      0.000000   
50%       37.000000  1.783560e+05      10.000000      0.000000      0.000000   
75%       48.000000  2.370510e+05      12.000000      0.000000      0.000000   
max       90.000000  1.484705e+06      16.000000  99999.000000   4356.000000   

            country  
count  32561.000000  
mean      40.437456  
std       12.347429  
min        1.000000  
25%       40.000000  
50%       40.000000  
75%       45.000000  
max       99.000000  

## 🧹 Data Cleaning & Quality Audit

In this phase, the dataset was cleaned by handling missing values, removing duplicates, standardizing categories, and detecting outliers. Basic transformations were applied to prepare the data for reliable analysis.

### 🔹 Code
import numpy as np
#Check missing values
print("Missing values per column:")
print(df.isnull().sum())

#Check duplicates
dup_count=df.duplicated().sum()
print("\nDuplicates rows:",dup_count)

#Standardize categories
for col in df.select_dtypes(include="object").columns:
  df[col]=df[col].astype(str).str.strip()

#Outlier detection
def iqr_rule(series):
  q1,q3 =series.quantile([0.25,0.75])
  iqr=q3-q1
  lower=q1-1.5*iqr
  upper=q3+1.5*iqr
  return(series<lower)|(series>upper)
outlier_summary={}
for col in df.select_dtypes(include={np.number}).columns:
  flags=iqr_rule(df[col])
  outlier_summary[col]=flags.sum()
  print("\nOutlier counts per numeric column:")
  print(outlier_summary)

#Simple transformations
df["log 1p_capital_gain"]=np.log1p(df["capital_gain"])
df["log 1p_capital_loss"]=np.log1p(df["capital_loss"])

print("\nNew transformed columns added:log1p_capital_gain,log1p_capital_loss")

Missing values per column:
age                  0
workclass         1836
fnlwgt               0
education            0
maritalstatus        0
occupation           0
education_num     1843
relationship         0
race                 0
gender               0
capital_gain         0
capital_loss         0
country              0
income             583
hours_per_week       0
dtype: int64

Duplicates rows: 24

Outlier counts per numeric column:
{'age': np.int64(143)}

Outlier counts per numeric column:
{'age': np.int64(143), 'fnlwgt': np.int64(992)}

Outlier counts per numeric column:
{'age': np.int64(143), 'fnlwgt': np.int64(992), 'maritalstatus': np.int64(1198)}

Outlier counts per numeric column:
{'age': np.int64(143), 'fnlwgt': np.int64(992), 'maritalstatus': np.int64(1198), 'capital_gain': np.int64(2712)}

Outlier counts per numeric column:
{'age': np.int64(143), 'fnlwgt': np.int64(992), 'maritalstatus': np.int64(1198), 'capital_gain': np.int64(2712), 'capital_loss': np.int64(1519)}

Outlier counts per numeric column:
{'age': np.int64(143), 'fnlwgt': np.int64(992), 'maritalstatus': np.int64(1198), 'capital_gain': np.int64(2712), 'capital_loss': np.int64(1519), 'country': np.int64(9008)}

New transformed columns added:log1p_capital_gain,log1p_capital_loss

## 📊 Exploratory Data Analysis (EDA)

In this phase, univariate, bivariate, and multivariate analyses were performed to uncover patterns in the dataset. Visualizations such as histograms, bar plots, and heatmaps were used to highlight key relationships, leading to 5–8 insights about factors influencing income.

import matplotlib.pyplot as plt
import seaborn as sns

#Univariate analysis(numeric distribution)

numeric_cols=["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
for col in numeric_cols:
  plt.figure(figsize= (6,4))
  sns.histplot(df[col], bins=30, kde=True)
  plt.title(f"Distribution of {col}")
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.show()

#Univariate analysis(categorical distribution)

cat_cols=[c for c in df.select_dtypes(include="object").columns if c!="income"]
for col in cat_cols:
  plt.figure()
  df[col].value_counts().head(10).plot(kind="bar")
  plt.title(f"Top categories in {col}")
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.xticks(rotation=45,ha="right")
  plt.show()

#Bivariate analysis(income vs categorical)
def income_rate_by(col):
  return df.groupby(col)["income"].apply(lambda x:(x== ">50K").mean()).sort_values(ascending=False)
  key_cats=["sex","education","maritalstatus","relationship","workclass","race","nativecountry","occupation"]
  for col in key_cats:
    rate=income_rate_by(col)
    plt.figure
    rate.plot(kind="bar")
    plt.title(f"Proportion earning >50k by {col}")
    plt.ylabel("Proportion >50k ")
    plt.xticks(rotation=45,ha="right")
    plt.show()

#Multivariate analysis

corr=df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(8,6))
plt.imshow(corr,cmap="coolwarm",aspect="auto")
plt.colorbar()
plt.title("Correlation heatmap")
plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.show()

![Age Distribution](https://github.com/user-attachments/assets/d9c64fd9-0b5b-4f25-876b-58a5928a0ab1)

![Income vs Education](https://github.com/user-attachments/assets/46b9139c-763f-45bf-b552-a409cfe46096)

![Income vs Occupation](https://github.com/user-attachments/assets/4bad9a8a-61dc-4f37-9085-5fb4a7741dcd)

![Income vs Gender](https://github.com/user-attachments/assets/d7819843-dfff-457c-9e07-34afb7e4e5c3)

![Correlation Heatmap](https://github.com/user-attachments/assets/b19d6949-6324-4413-a63e-8d2fbccb1404)

![Boxplot of Hours Worked](https://github.com/user-attachments/assets/0b73c658-a042-49dd-b5bd-ee2c3e517c16)

![Capital Gain Distribution](https://github.com/user-attachments/assets/112b1e41-ffca-4a8f-ad9b-721b12876631)

![Capital Loss Distribution](https://github.com/user-attachments/assets/260455ad-5f03-49cd-b845-4d5bcdc8aedd)

![Hours per Week Distribution](https://github.com/user-attachments/assets/6a7105bf-b333-4557-946a-c4a90cb3ab33)

![Pairplot](https://github.com/user-attachments/assets/d921c20f-78c8-4440-92e2-ee7f95e5c3c0)
