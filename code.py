import statsmodels as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = "car_price_data.csv"
df = pd.read_csv(data,header=None)
print("Dataset Import Successful")
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("Header Load Successful")
df.columns=headers
import numpy as np
df.replace("?", np.nan, inplace = True)
df.head(5)
missing_data = df.isnull()
missing_data.head(5)
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
avg_bore=df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)
avg_stroke=df["stroke"].astype("float").mean(axis=0)
df["stroke"].replace(np.nan,avg_stroke,inplace=True)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan,'four',inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(3)
df.dtypes
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.to_csv("cleaned_data.csv",index=False)
df["city-L/100km"]=235/df["city-mpg"]
df.head(1)
df["highway-mpg"]=235/df["highway-mpg"]
df.rename(columns={"highway-mpg":"highway-L/100km"},inplace=True)
df.head(1)
df["length"]=df["length"]/df["length"].max()
df["width"]=df["width"]/df["width"].max()
df["height"]=df["height"]/df["height"].max()
df["horsepower"].nunique()
df["horsepower"]=df["horsepower"].astype(int,copy=True)
plt.hist(df["horsepower"])
plt.xlabel("HORSEPOWER")
plt.ylabel("COUNT")
plt.title("HISTOGRAM FOR HORSEPOWER VARIABLE")
plt.show()
bins = np.linspace(min(df["horsepower"]),max(df["horsepower"]),4)
bins
names = ['low','medium','high']
df["horsepower-binned"] = pd.cut(df["horsepower"],bins,labels=names,include_lowest=True)
df[["horsepower","horsepower-binned"]].head()
df["horsepower-binned"].value_counts()
plt.hist(df["horsepower-binned"])
plt.xlabel("HORSEPOWER-BINNED")
plt.ylabel("COUNT")
plt.title("DISTRIBUTION OF BINNED HORSEPOWER")
plt.show()
plt.hist(df["horsepower"],bins=3)
plt.xlabel("HORSEPOWER-BINNED")
plt.ylabel("COUNT")
plt.title("DISTRIBUTION OF BINNED HORSEPOWER")
plt.show()
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'gas':'fuel-type-gas','diesel':'fuel-type-diesel'},inplace=True)
dummy_variable_1.head()
df=pd.concat([df,dummy_variable_1],axis=1)
df.drop("fuel-type",axis=1,inplace=True)
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.head()
dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'},inplace=True)
dummy_variable_2.head()
df= pd.concat([df,dummy_variable_2],axis=1)
df.drop("aspiration",axis=1,inplace=True)
df.columns
df.to_csv('clean_and_binned.csv')


















