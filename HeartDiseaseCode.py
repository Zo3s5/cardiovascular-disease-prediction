import pandas as pd # loads the dataaset which saved in the same folder as this jupyter ipynb file

df = pd.read_csv('C_disease.csv') 

print(df)  # prints out the the first 5 rows of each column 

df.head() # prints out the dataset in a different format showing the columns and rows 

import pandas as pd

df = pd.read_csv('C_disease.csv')
 # checks the dataset for any duplicated values 
duplicated_rows = df.duplicated() 
df = duplicated_rows.sum()  # calculates the total sum of the number of duplicated values 

print(f'There are {df} duplicated rows.')

import pandas as pd

df = pd.read_csv('C_disease.csv')

print(df.isna().any()) # prints if there is any null values in the columns 


import pandas as pd

df = pd.read_csv('C_disease.csv')

numof0 = df['cardio'].value_counts().get(0, 0)  # counts the number of values of 0s in cardio column 
numof1 = df['cardio'].value_counts().get(1, 0)  # counts the number of values of 0s in cardio column

print("Number of 0s:", numof0)
print("Number of 1s:", numof1)


import pandas as pd
import numpy as np

df = pd.read_csv('C_disease.csv')

df = pd.DataFrame(df)
  
# using isnull() function  
df.isnull() 

import pandas as pd

df = pd.read_csv('C_disease.csv')

print(df.mean(numeric_only=True))   # calculates the means 


import pandas as pd

df = pd.read_csv('C_disease.csv')

print(df.median(numeric_only=True))    # calculates the mmedians  



import pandas as pd

df = pd.read_csv('C_disease.csv')

print(df.mode(numeric_only=True))    # calculates the modes 




import pandas as pd

df = pd.read_csv('C_disease.csv')

result = df[df['age'] == df['age'].max()]   #prints out the reslts with the highest age 

print(result)


# Data Preprocessing Dataset 1

import pandas as pd
import numpy as np

df = pd.read_csv('C_disease.csv')
df['age'] = df['age'] // 365   #round down age from days to years 


print(df)
df.to_csv('C_disease.csv', index=False) #saves the changes                




import pandas as pd

df = pd.read_csv('C_disease.csv')

number_of_smokers = df[df['smoke'] == 1].shape[0]  #prints out the total number of smokers 

print(number_of_smokers)


import pandas as pd

df = pd.read_csv('C_disease.csv')

female_smokers = df[(df['gender'] == 1) & (df['smoke'] == 1)].shape[0]   # prints out the number of female smokers 

male_smokers = df[(df['gender'] == 2) & (df['smoke'] == 1)].shape[0]

print("Number of female smokers:", female_smokers)
print("Number of male smokers:", male_smokers) # prints out the number of male smokers 

import pandas as pd

df = pd.read_csv('C_disease.csv')

number_of_alcoholics = df[df['alco'] == 1].shape[0]

print(number_of_alcoholics)   # prints ou the total number of alcoholics 


import pandas as pd

df = pd.read_csv('C_disease.csv')

female_drinkers = df[(df['gender'] == 1) & (df['alco'] == 1)].shape[0]

male_drinkers = df[(df['gender'] == 2) & (df['alco'] == 1)].shape[0]

print("Number of female drinkers:", female_drinkers)   # prints out the number of female drinkers/alcoholics
print("Number of male drinkers:", male_drinkers) # prints out the number of male drinkers/alcoholics

import pandas as pd

df = pd.read_csv('C_disease.csv')

df['height'] = df['height'] / 100  
df['BMI'] = df['weight'] / (df['height'] ** 2)
df['BMI'] = df['BMI'].round(1)              # calculates the bmi by using the height and weight in the dataset 

print(df.head())


df.to_csv('updated1_dataset.csv', index=False)  #saves it to a new file 



# Data Exploration Dataset 2

import pandas as pd

df = pd.read_csv('h_disease.csv')

numof0 = df['HeartDiseaseorAttack'].value_counts().get(0, 0)  # uses .value.count function 
numof1 = df['HeartDiseaseorAttack'].value_counts().get(1, 0)

print("Number of 0s:", numof0)     # prints out the number of patients with HeartDiseaseorAttack 
print("Number of 1s:", numof1)      # prints out the number of patients with HeartDiseaseorAttack 

import pandas as pd

df = pd.read_csv('h_disease.csv')

print(df)

import pandas as pd

df = pd.read_csv('h_disease.csv')

duplicated_rows = df.duplicated()
df = duplicated_rows.sum()                # prints the number of duplicated rows

print(f'There are {df} duplicated rows.')

import pandas as pd

df = pd.read_csv('h_disease.csv')

print(df.isna().any()) # looks for any missing values 


import pandas as pd

df = pd.read_csv('h_disease.csv')

result = df[df['Age'] == df['Age'].max()]

print(result)         # prints all the rows where he maximum age is group 13

import pandas as pd

df = pd.read_csv('h_disease.csv')

print(df.mean(numeric_only=True))  # prints the means of each column 

import pandas as pd

df = pd.read_csv('h_disease.csv')

print(df.median(numeric_only=True))     # prints the median  of each column 


import pandas as pd

df = pd.read_csv('h_disease.csv')

print(df.mode(numeric_only=True))        # prints the mode of each column 











# Data Preprocessing Dataset 2

import pandas as pd

df = pd.read_csv('h_disease.csv')

df_duplicatedrow = df.drop_duplicates()  # finds all the duplicated values and removes them using the df.drop function 

print(df_duplicatedrow)

df_duplicatedrow.to_csv('updated_dataset2.csv', index=False)   # is saved to a new file 

import pandas as pd

df = pd.read_csv('updated_dataset2.csv')

duplicated_rows = df.duplicated()
df = duplicated_rows.sum()      # prints the number of duplicated rows

print(f'There are {df} duplicated rows.')

import pandas as pd

df = pd.read_csv('updated_dataset2.csv')

df['TotalHlth'] = df['PhysHlth'] + df['MentHlth']   # cobines physhlth and menhlth to a new column TotalHlth

df.drop(columns=['PhysHlth', 'MentHlth'], inplace=True)  # removes the two columns 

print(df.head())







# Data Exploration Dataset 3 

import pandas as pd

df = pd.read_csv('heart_RP.csv')

print(df)

import pandas as pd

df = pd.read_csv('heart_RP.csv')

numof0 = df['condition'].value_counts().get(0, 0)
numof1 = df['condition'].value_counts().get(1, 0)

print("Number of 0s:", numof0)           # prints out the number of patients with condition
print("Number of 1s:", numof1)           # prints out the number of patients with condition 


import pandas as pd

df = pd.read_csv('heart_RP.csv')

df_duplicatedrow = df.drop_duplicates()

print(df_duplicatedrow)     # removes duplicated rows and file is saved to a new file 

df_duplicatedrow.to_csv('updated_dataset3.csv', index=False)

import pandas as pd

df = pd.read_csv('heart_RP.csv')

duplicated_rows = df.duplicated()  # calculates the total sum of duplicated rows 
df = duplicated_rows.sum()

print(f'There are {df} duplicated rows.')

import pandas as pd

df = pd.read_csv('heart_RP.csv')

print(df.isna().any())    # finds any missing values 

import pandas as pd

df = pd.read_csv('heart_RP.csv')

print(df.mean(numeric_only=True))   # prints mean of each column 

import pandas as pd

df = pd.read_csv('heart_RP.csv')

print(df.mode(numeric_only=True))          # prints mode of each column

# Data Preprocessing Dataset 3

import pandas as pd

df = pd.read_csv('heart_RP.csv')

print(df)      # prints out the first five rows of the dataset 

import pandas as pd

df = pd.read_csv('heart_RP.csv')

duplicated_rows = df.duplicated()
df = duplicated_rows.sum()          # calculates the number od duplicated rows 

print(f'There are {df} duplicated rows.')

import pandas as pd

df = pd.read_csv('heart_RP.csv')
df_end = pd.get_dummies(df, columns=['cp', 'slope', 'restecg', 'thal'], prefix=['cp', 'slope', 'restecg', 'thal'])
# one hot encoding of 4 columns which has multiple values 
df_end.to_csv('updated3_dataset.csv', index=False)   
print(df_end)



# Data Visualisation Dataset 1 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('updated1_dataset.csv')

# Age distribution of patients       
sns.histplot(data['age'], bins=30)
plt.title('Age Distribution of Patients')
plt.show()  # histogram 


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('updated1_dataset.csv')


df = pd.DataFrame(df)

bins = [35, 40, 45, 50, 55, 60, 65]
labels = ["35-40","40-45", "45-50", "50-55", "55-60", "60-65"]        # age range 
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
grouped = df.groupby('AgeGroup')['cholesterol'].sum()    # calculates number of people with cholesterol in each age range 

plt.bar(grouped.index, grouped.values, color=['orange', 'purple','red','blue','silver','yellow'])
plt.xlabel('Age Group')
plt.ylabel('Number of Cholesterol Cases')
plt.title('Cholesterol Cases by Age Group')
for age_group, total_cases in grouped.items():     #  prints out the total number of patients with cholesterol 
    print(f"At Age Group: {age_group},The Total Cholesterol Cases: {total_cases}")

plt.show()

import matplotlib.pyplot as plt #plotting & visualization.
import seaborn as sns 


df = pd.read_csv('updated1_dataset.csv')

corr = df.corr() 
corr
                                           #  creates a heatmap 
plt.subplots(figsize=(14,14))     

sns.heatmap(corr,cmap= 'RdYlGn',annot=True)
plt.show() 





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('updated1_dataset.csv')


data = pd.DataFrame(df)


age_bins = [data['age'].min(), 30, 40, 50, 60, data['age'].max()]
age_labels = ['<30', '30-40', '40-50', '50-60', '60+']   # age range 
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

for label in age_labels:
    group = df[df['age_group'] == label]['height']
    Q1, Q2, Q3 = group.quantile([0.25, 0.5, 0.75])
    IQR = Q3 - Q1     # boxplot of the 3 quartiles 
   
    print(f"Age Group: {label}")
    print(f" Q1:{Q1},  Median:{Q2},  Q3:{Q3}")   # prints the information of the different age groups 

plt.figure(figsize=(12, 8))
sns.boxplot(x='age_group', y='height', data=data, palette="husl")
plt.title('Box Plot of Cholesterol Levels by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Cholesterol Level (mg/dL)')


plt.show() # creates a boxplot 










# Data Visualisation 2

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('updated_dataset2.csv')

plt.figure(figsize=(10, 5))
sns.histplot(df['BMI'], kde=True)
plt.title('Histogram of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()                      # prints the histogram of the BMI 


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('h_disease.csv')   

# Visualization: Age distribution of patients
sns.histplot(data['Age'], bins=30)
plt.title('Age Distribution of Patients')
plt.show() # prints the histogram 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('updated_dataset2.csv')

sns.set(style="whitegrid")    # bar graph showing trends o 

plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='HeartDiseaseorAttack', data=data, palette="Blues_d")
plt.title('Prevalence of Heart Disease or Attack by Age Group')
plt.xlabel('Age Group')            # prints the 14 level age categrory 
plt.ylabel('Proportion with Heart Disease or Attack')  # prints the 14 level age categrory with those who have heart disease 
                                                       #or heart attack  
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Stroke', data=data, palette="Greens_d")
plt.title('Prevalence of Stroke by Age Group')
plt.xlabel('Age Group')  # prints the 14 level age categrory 
plt.ylabel('Proportion with Stroke') # prints the 14 level age categrory with those who have stroke 
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Diabetes', data=data, palette="Reds_d")
plt.title('Prevalence of Diabetes by Age Group')
plt.xlabel('Age Group')  # prints the 14 level age categrory 
plt.ylabel('Proportion with Diabetes') # prints the 14 level age categrory with those who have diabetes 
plt.show()


# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Reading the data from a CSV file
data = pd.read_csv('h_disease.csv')

# Defining the lifestyle factors and health outcome
lifestyle_factors = ['PhysActivity', 'Fruits', 'Smoker']
health_outcome = 'HeartDiseaseorAttack'

# Grouping data by lifestyle factors and calculating the mean of health outcome
lifestyle_impact = data.groupby(['Smoker', 'PhysActivity', 'Fruits'])[health_outcome].mean().reset_index()

# Setting the style of the plots
sns.set(style="whitegrid")

# Creating a bar plot to show the impact of Physical Activity and Smoking on Heart Disease
plt.figure(figsize=(12, 8))
sns.barplot(x='PhysActivity', y=health_outcome, hue='Smoker', data=lifestyle_impact)
plt.title('Impact of Physical Activity and Smoking on Heart Disease')
plt.xlabel('Physical Activity')
plt.ylabel('Proportion with Heart Disease')
plt.legend(title='Smoker')
plt.show()

# Creating a bar plot to show the impact of Fruit Consumption and Smoking on Heart Disease
plt.figure(figsize=(12, 8))
sns.barplot(x='Fruits', y=health_outcome, hue='Smoker', data=lifestyle_impact)
plt.title('Impact of Fruit Consumption and Smoking on Heart Disease')
plt.xlabel('Fruit Consumption')
plt.ylabel('Proportion with Heart Disease')
plt.legend(title='Smoker')
plt.show()

import matplotlib.pyplot as plt # Importing matplotlib for plotting and visualization.
import seaborn as sns # Importing seaborn for visualization package

df = pd.read_csv('h_disease.csv') # Reading the data from 'h_disease.csv' file.

corr = df.corr() # Calculating the correlation matrix.

plt.subplots(figsize=(14,14)) # Setting the size of the plot.

sns.heatmap(corr,cmap= 'RdYlGn',annot=True) # Creating a heatmap of the correlation matrix with color map 'RdYlGn' and annotations.

plt.show() # Display the heatmap plot.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from 'h_disease.csv' file
df = pd.read_csv('h_disease.csv')

# Ensure 'Age' is used for binning and analysis
# Adjust max age for unique bin edge
min_age, max_age = df['Age'].min(), df['Age'].max() + 1
age_bins = [min_age, 3, 6, 9, 13, max_age]
age_labels = ['<1', '1-4', '5-8', '9-12', '13+']
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate and print quartiles for each age group for BMI
for label in age_labels:
    group = df[df['age_group'] == label]['Income']
    Q1, Q2, Q3 = group.quantile([0.25, 0.5, 0.75])
    print(f"Age Group: {label}")
    print(f" Q1:{Q1},  Median:{Q2},  Q3:{Q3}")

# Plotting BMI by age group
plt.figure(figsize=(12, 8))
sns.boxplot(x='age_group', y='Income', data=df, palette="husl")
plt.title('Box Plot of BMI by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Income')
plt.show()










# Data Visualisation Dataset 3 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('heart_RP.csv')

sns.histplot(data['age'], bins=30) # Creating a histogram plot of age with 30 bins.
plt.title('Age Distribution of Patients') # Setting the title of the plot.
plt.show() # Display the histogram plot.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loads the dataset from 'heart_RP.csv' file
df = pd.read_csv('heart_RP.csv')

# Sets the style of the visualization to 'whitegrid'
sns.set_style("whitegrid")

# Creatse a figure with a specific size
plt.figure(figsize=(10, 6))

# Creates a scatter plot 
sns.scatterplot(x='age', y='chol', data=df, color='blue', alpha=0.6)

# Adds title and labels with specific font sizes
plt.title('Cholesterol Level vs Age', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Cholesterol Level (mg/dL)', fontsize=12)


plt.show()

import pandas as pd


df = pd.read_csv('heart_RP.csv')

# Defines the  age ranges and labels
bins = [29, 39, 49, 59, 69, 79]
labels = ['30-39', '40-49', '50-59', '60-69', '70-79']
df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Calculate the average trestbps for each age range
age_grouped_avg_trestbps = df.groupby('age_range')['trestbps'].mean().reset_index()

# Prints the averages
print(age_grouped_avg_trestbps)


import matplotlib.pyplot as plt #plotting & visualization.
import seaborn as sns #visualization package


df = pd.read_csv('heart_RP.csv')

corr = df.corr() 
corr

plt.subplots(figsize=(14,14))

sns.heatmap(corr,cmap= 'RdYlGn',annot=True)
plt.show()    # prints the heatmap 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart_RP.csv')


data = pd.DataFrame(df)


age_bins = [data['age'].min(), 40, 50, 60, 70, data['age'].max()]
age_labels = ['<40', '40-49', '50-59', '60-69', '70+']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

for label in age_labels:
    group = df[df['age_group'] == label]['chol']
    Q1, Q2, Q3 = group.quantile([0.25, 0.5, 0.75])
    IQR = Q3 - Q1    # Calculates the quartile values and IQR for cholesterol levels in each age group.
   
    print(f"Age Group: {label}")
    print(f" Q1:{Q1},  Median:{Q2},  Q3:{Q3}")

plt.figure(figsize=(12, 8))
sns.boxplot(x='age_group', y='chol', data=data, palette="husl")
plt.title('Box Plot of Cholesterol Levels by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Cholesterol Level (mg/dL)')  #  box plot of cholesterol levels by age group


plt.show() 



# Machine Learning Models for Dataset 1 

#Decision Tree 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score


df = pd.read_csv('updated1_dataset.csv')


X = df.drop('cardio', axis=1)  
y = df['cardio'] 
# Splits the dataset into features (X) and target variable (y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)


dtc = DecisionTreeClassifier()
#Decision Tree classifier to the train data 

dtc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

prediction = dtc.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

# calculates the metrics 

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")




#Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('updated_dataset.csv')


X = df.drop('cardio', axis=1)  
y = df['cardio']   

# Splits the dataset into features (X) and target variable (y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)


lr = LogisticRegression(max_iter=10000) 


lr.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

prediction = lr.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

# performance metrics


print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



#XGBoost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

df = pd.read_csv('updated_dataset.csv')

X = df.drop('cardio', axis=1)  
y = df['cardio']  
# Splits the dataset into features (X) and target variable (y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Calculating various metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)


# Printing the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


#Random Forest Dataset 1 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier # imports the RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('updated_dataset.csv')

X = df.drop('cardio', axis=1)  
y = df['cardio']

# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

# trains and creates RandomForestClassifier
rfc = RandomForestClassifier() 
rfc.fit(X_train, y_train)

prediction = rfc.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_pred)



print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



#Random Forest Dataset 1 using SMOTE


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('updated_dataset.csv')


X = df.drop('cardio', axis=1)
y = df['cardio']

# Splits the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

# adding SMOTE to the training data to improve the imbalance 
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Training the RandomForestClassifier on the SMOTE-applied training data
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# Making predictions
prediction = rfc.predict(X_test)

# performance metrics
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_pred)


print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


#Random Forest Dataset 1 Balanced

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv('updated_dataset.csv')

X = df.drop('cardio', axis=1)
y = df['cardio']
# Splits the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

# Uses Balanced Random Forest Classifier
brf_classifier = BalancedRandomForestClassifier(random_state=42, n_estimators=100)
brf_classifier.fit(X_train, y_train)

y_pred = brf_classifier.predict(X_test)

# performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)



print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('updated_dataset.csv')
X = df.drop(columns=['cardio'])
y = df['cardio']

# Split the dataset into train, validate, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1000)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1000)

# Standardises the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# training the model 
model = create_model(X_train.shape[1])

#early stopping to prevent overfitting 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}')
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred)}")

# keras model 
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

#  cross-validation is executed 
clf = KerasClassifier(model)
scores = cross_val_score(clf, X_train, y_train, cv=5)

# cross-validation results
plt.figure(figsize=(8, 6))
plt.hist(scores, bins=5, color='blue', alpha=0.7)
plt.title('Model Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Average Cross-Validation Accuracy: {np.mean(scores)}")












# Machine Learning Models for Dataset 2 

#Decision Tree Dataset 2


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv('updated_dataset2.csv')
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# Splits the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)
#Decision Tree classifier to the train data 
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

prediction = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

# performance metrics

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



# Decision Tree Dataset 2 with SMOTE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)  
y = df['HeartDiseaseorAttack']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

smote = SMOTE(random_state=100000)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Splits the dataset into training and testing sets


# adding SMOTE to the training data to improve the imbalance 

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_smote, y_train_smote)

prediction = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")  
print(f"AUC Score: {auc}")



#Logisitic Regression Dataset 2 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)  
y = df['HeartDiseaseorAttack']  
# Splits the dataset into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

# runs the logistic regression model 
lr = LogisticRegression(max_iter=10000) 


lr.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

prediction = lr.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

# performance metrics


print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



# Logistic Regression Dataset 2 with SMOTE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# Splits the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

smote = SMOTE(random_state=100000)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# adding SMOTE to the training data to improve the imbalance 

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_smote, y_train_smote)
# runs the logistic regression model 
y_prediction = lr.predict(X_test)
prediction = lr.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



#XGBoost Dataset 2 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)  # Features
y = df['HeartDiseaseorAttack']  # Target variable

# Splits the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)
# the classifier to the training data to build the XGBoost model.
xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

y_prediction = xgb_classifier.predict(X_test)
prediction = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics


print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



# XGBosst Dataset 2 with SMOTE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)  # Features
y = df['HeartDiseaseorAttack']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

# Applying SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

## the classifier to the training data to build the XGBoost model.

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_smote, y_train_smote)

y_prediction = xgb_classifier.predict(X_test)
prediction = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



#Random Forest Dataset 2

import pandas as pd
from sklearn.ensemble import RandomForestClassifier # imports the RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)  
y = df['HeartDiseaseorAttack']

# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)

# trains and creates RandomForestClassifier
rfc = RandomForestClassifier() 
rfc.fit(X_train, y_train)

y_prediction = rfc.predict(X_test)
prediction = rfc.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")




# Random Forest Dataset 2 with SMOTE 

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

# adding SMOTE to the training data to improve the imbalance 

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Training the RandomForestClassifier on the SMOTE-applied training data

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

y_prediction = rfc.predict(X_test)
prediction = rfc.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


# Random Forest Dataset 2 Balanced 

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

df = pd.read_csv('updated_dataset2.csv')

X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']
# Splits the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100000)
# the classifier to the training data to build the BalancedRandomForest model.

brf_classifier = BalancedRandomForestClassifier(random_state=42, n_estimators=100)
brf_classifier.fit(X_train, y_train)

prediction = brf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, prediction)  
# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv('updated_dataset2.csv')
X = df.drop(columns=['HeartDiseaseorAttack'])
y = df['HeartDiseaseorAttack']

# Splits the dataset into training and testing sets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1000)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1000)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model
model = create_model(X_train.shape[1])
# early stopping to prevent overfitting 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# performence testing 
y_pred = (model.predict(X_test) > 0.5).astype(int)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}')
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred)}")

# Integrates Keras model with scikit-learn
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

# Performs cross-validation
clf = KerasClassifier(model)
scores = cross_val_score(clf, X_train, y_train, cv=5)

# cross-validation results

plt.figure(figsize=(8, 6))
plt.hist(scores, bins=5, color='blue', alpha=0.7)
plt.title('Model Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Average Cross-Validation Accuracy: {np.mean(scores)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('updated_dataset2.csv')
X = df.drop(columns=['HeartDiseaseorAttack'])
y = df['HeartDiseaseorAttack']

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1000)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1000)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Apply SMOTE to address class imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Define a function to create the model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model
model = create_model(X_train.shape[1])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# metrics
y_pred = (model.predict(X_test) > 0.5).astype(int)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}')
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred)}")

# keras model 
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

#  cross-validation
clf = KerasClassifier(model)
scores = cross_val_score(clf, X_train, y_train, cv=5)

# cross-validation results
plt.figure(figsize=(8, 6))
plt.hist(scores, bins=5, color='blue', alpha=0.7)
plt.title('Model Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Average Cross-Validation Accuracy: {np.mean(scores)}")












# Machine Learning Models for Dataset 3

# Decision Tree Dataset 3  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


df = pd.read_csv('updated3_dataset.csv')


X = df.drop('condition', axis=1)  
y = df['condition'] 

# Splits the dataset into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# the classifier to the training data to build the  DecisionTreeClassifier

dtc = DecisionTreeClassifier()


dtc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = dtc.predict(X_test)

predictions = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")



# Logisitic Regression LDataset 3 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('updated3_dataset.csv')

X = df.drop('condition', axis=1)  
y = df['condition']  

# Splits the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# the classifier to the training data to build the LogisticRegression model.

lr = LogisticRegression(max_iter=10000) 


lr.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = dtc.predict(X_test)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")


# XGBoost Dataset 3 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb 
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv('updated3_dataset.csv')

X = df.drop('condition', axis=1)  
y = df['condition']  # Target 

# Splits the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)
# the classifier to the training data to build the XGBoost model.

y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")

#Random Forest Dataset 3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier # imports the RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,accuracy_score


df = pd.read_csv('updated3_dataset.csv')

X = df.drop('condition', axis=1)  
y = df['condition']

# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# trains and creates RandomForestClassifier
rfc = RandomForestClassifier() 
rfc.fit(X_train, y_train)


y_prediction = rfc.predict(X_test)
prediction = rfc.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, y_prediction)

# performance metrics

print(f"Accuracy: {accuracy}")  
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")




# Multi layer Percepton Dataset 3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('updated3_dataset.csv')

X = df.drop('condition', axis=1)
y = df['condition']


# Split the dataset into train, validate, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=100)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=100)


continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Standardises only the continuous columns
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# nueral network 
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# trains the model 
model = create_model(X_train.shape[1])
# early stoppping to prevent overfitting 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# metrics 
y_pred = (model.predict(X_test) > 0.5).astype(int)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}')
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred)}")

# Integrates the  Keras model 
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

# cross_validation happens 
clf = KerasClassifier(model)
scores = cross_val_score(clf, X_train, y_train, cv=5)

#  diagram plot 
plt.figure(figsize=(8, 6))
plt.hist(scores, bins=5, color='blue', alpha=0.7)
plt.title('Model Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

print(f"Average Cross-Validation Accuracy: {np.mean(scores)}")











































# Data Visualisation for Machine Learning Models

import matplotlib.pyplot as plt
import numpy as np

models_dataset_1 = [
    "Decision Tree", "Logistic Regression", "XGBoost",
    "Random Forest", "MLP"  
]

# metric scores 
accuracy_1 = [0.63, 0.71, 0.74, 0.72, 0.72]
precision_1 = [0.63, 0.74, 0.75, 0.73, 0.72]
recall_1 = [0.63, 0.65, 0.71, 0.70, 0.72]
f1_score_1 = [0.63, 0.69, 0.73, 0.72, 0.72]
AUC_score_1 = [0.63, 0.71, 0.74, 0.50, 0.72]

# settings of the plot size 
fig, ax = plt.subplots(figsize=(12, 7))

index_1 = np.arange(len(models_dataset_1))
bar_width = 0.15 
opacity = 0.8

# creation of the bars 
bar1 = ax.bar(index_1 - bar_width*2, accuracy_1, bar_width, alpha=opacity, label='Accuracy')
bar2 = ax.bar(index_1 - bar_width, precision_1, bar_width, alpha=opacity, label='Precision')
bar3 = ax.bar(index_1, recall_1, bar_width, alpha=opacity, label='Recall')
bar4 = ax.bar(index_1 + bar_width, f1_score_1, bar_width, alpha=opacity, label='F1 Score')
bar5 = ax.bar(index_1 + bar_width*2, AUC_score_1, bar_width, alpha=opacity, label='AUC Score')

# adding titles 
ax.set_xlabel('Models')
ax.set_ylabel('Scores')   
ax.set_title('Performance of Machine Learning Models on Dataset 1')
ax.set_xticks(index_1)
ax.set_xticklabels(models_dataset_1, rotation=45) 
ax.legend()

plt.tight_layout()
plt.show()











import matplotlib.pyplot as plt
import numpy as np


models_dataset_2 = [
    "Decision Tree", "Decision Tree with SMOTE", "Logistic Regression",
    "Logistic Regression with SMOTE", "XGBoost", "XGBoost with SMOTE",
    "Random Forest", "Balanced Random Forest", "MLP", "MLP With Smote"
]

# metric scores 
accuracy_2 = [0.83, 0.76, 0.90, 0.71, 0.90, 0.72, 0.89, 0.72, 0.72, 0.77]
precision_2 = [0.23, 0.18, 0.53, 0.22, 0.51, 0.21, 0.40, 0.24, 0.73, 0.61]
recall_2 = [0.28, 0.37, 0.13, 0.73, 0.12, 0.66, 0.12, 0.80, 0.72, 0.71]
f1_score_2 = [0.25, 0.24, 0.20, 0.33, 0.20, 0.32, 0.17, 0.36, 0.72, 0.62]
AUC_2 = [0.58, 0.59, 0.56, 0.72, 0.55, 0.69, 0.54, 0.75, 0.72, 0.71]

# settings of the plot size 
fig, ax = plt.subplots(figsize=(15, 7))

index_2 = np.arange(len(models_dataset_2))
bar_width = 0.15 
opacity = 0.8
# creation of the bars 

bar1 = ax.bar(index_2 - bar_width*2, accuracy_2, bar_width, alpha=opacity, label='Accuracy')
bar2 = ax.bar(index_2 - bar_width, precision_2, bar_width, alpha=opacity, label='Precision')
bar3 = ax.bar(index_2, recall_2, bar_width, alpha=opacity, label='Recall')
bar4 = ax.bar(index_2 + bar_width, f1_score_2, bar_width, alpha=opacity, label='F1 Score')
bar5 = ax.bar(index_2 + bar_width*2, AUC_2, bar_width, alpha=opacity, label='AUC')

# adding titles 
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance of Machine Learning Models on Dataset 2')
ax.set_xticks(index_2)
ax.set_xticklabels(models_dataset_2, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()










import matplotlib.pyplot as plt
import numpy as np

models_dataset_3 = [
    "Decision Tree", "Logistic Regression", "XGBoost",
    "Random Forest", "MLP"
]

# metric scores 
accuracy_3 = [0.79, 0.79, 0.83, 0.83, 0.89]
precision_3 = [0.84, 0.84, 0.91, 0.91, 0.89]
recall_3 = [0.76, 0.76, 0.78, 0.78, 0.91]
f1_score_3 = [0.80, 0.80, 0.84, 0.84, 0.89]
AUC_3 = [0.80, 0.80, 0.84, 0.84, 0.91]

# settings of the plot size fig, ax = plt.subplots(figsize=(12, 7))

index_3 = np.arange(len(models_dataset_3))
bar_width = 0.15 
opacity = 0.8
# creation of the bars 

bar1 = ax.bar(index_3 - bar_width*2, accuracy_3, bar_width, alpha=opacity, label='Accuracy')
bar2 = ax.bar(index_3 - bar_width, precision_3, bar_width, alpha=opacity, label='Precision')
bar3 = ax.bar(index_3, recall_3, bar_width, alpha=opacity, label='Recall')
bar4 = ax.bar(index_3 + bar_width, f1_score_3, bar_width, alpha=opacity, label='F1 Score')
bar5 = ax.bar(index_3 + bar_width*2, AUC_3, bar_width, alpha=opacity, label='AUC')

# adding titles 
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance of Machine Learning Models on Dataset 3')
ax.set_xticks(index_3)
ax.set_xticklabels(models_dataset_3, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()




