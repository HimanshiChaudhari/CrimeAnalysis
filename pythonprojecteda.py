#importig  Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

#IMPORTING DATASET
df = pd.read_csv("H:\\Kali Files\\Download\\Crime_Data_from_2020_to_Present.csv")
#DATASET INFO
print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
# Checking for missing values in the dataset and their total count
print(df.isnull().sum())

# Viewing the max,MIN,median,mean,modeand count values in the dataset
print(df.max(numeric_only=True))
print(df.min(numeric_only=True))
print(df.median(numeric_only=True))
print(df.mean(numeric_only=True))
print(df.mode(numeric_only=True))
print(df.count())

# Cleaning the dataset by dropping rows with missing values
print(df.dropna(inplace=True))
# Creating a numpy array from the crime rate
crime_code_array = np.array(df["Crm Cd"])
print(crime_code_array)
# Filtering years with crime data greater than 50
high_crime_years = df[df["Crm Cd"] > 50]
print(high_crime_years)
# Creating a histogram for the "Crm Cd" column
plt.hist(df["Crm Cd"], bins=10, color="blue", edgecolor="black")
plt.xlabel("Crime Code")
plt.ylabel("Frequency")
plt.title("Distribution of Crime Codes")
plt.show()
# Creating a bar chart to show the average crime code by area
avg_crime_code_by_area = df.groupby("AREA NAME")["Crm Cd"].mean()
avg_crime_code_by_area.plot(kind='bar', color='orange')
plt.xlabel("Area Name")
plt.ylabel("Average Crime Code")
plt.title("Crime Code by Area Name")
plt.show()
# Creating a line graph to show the trend of crime code across dates
plt.plot(df["DATE OCC"], df["Crm Cd"], marker='o')
plt.xlabel("Date of Occurrence")
plt.ylabel("Crime Code")
plt.title("Trend of Crime Code Across Dates")
plt.show()
#Scatter plot between 'Crm Cd' and 'Vict Age'
plt.scatter(df["Crm Cd"], df["Vict Age"], color='red')
plt.xlabel("Crime Code")
plt.ylabel("Victim Age")
plt.title("Crime Code vs Victim Age")
plt.show()

#Boxplot for "Crm Cd" distribution by year

sns.boxplot(x="Date Rptd", y="Crm Cd", data=df)
plt.title("Crime Code Distribution by Year")
plt.show()

#Creating a heatmap to visualize the correlation between features
numeric_df=df.select_dtypes(include=['number'])
numeric_df=numeric_df.dropna(axis=1,how='all')
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pair plot for selected columns
sns.pairplot(df[["Crm Cd", "Vict Age", "Date Rptd"]])
plt.show()
#creating Outlier
Q1=df['LAT'].quantile(0.25)
Q3=df['LAT'].quantile(0.75)
print(Q1)
print(Q3)
IQR=Q3-Q1
print(IQR)
lower_bound=Q1-1.5*IQR
print(lower_bound)
upper_bound=Q3+1.5*IQR
print(upper_bound)
outlier=df[(df['LAT']<lower_bound)| (df['LAT']>upper_bound)]
print(outlier)







