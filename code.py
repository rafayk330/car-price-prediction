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
df["peak-rpm"].dtype
for_correlation = df[["bore", "stroke","compression-ratio", "horsepower"]]
for_correlation.corr()
x = df["engine-size"]
y = df["price"]
plt.scatter(x,y)
plt.title("ENGINE-SIZE VS PRICE")
plt.xlabel("ENGINE-SIZE")
plt.ylabel("PRICE")
plt.show()
sns.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,)
df[["engine-size","price"]].corr()
df["highway-L/100km"].dtype
df[["highway-L/100km","price"]].corr()
sns.regplot(x="highway-L/100km",y="price",data=df)
plt.ylim(0,)
df[["peak-rpm","price"]].corr()
sns.regplot(x='peak-rpm',y='price',data=df)
plt.ylim(0,)
df[["stroke","price"]].corr()
sns.regplot(x="stroke",y="price",data=df)
plt.ylim(0,) 
sns.boxplot(x="engine-location",y="price",data=df)
sns.boxplot(x="body-style",y="price",data=df)
sns.boxplot(x="drive-wheels",y="price",data=df)
df.describe(include=['object'])
df.describe(include='all')
df["drive-wheels"].value_counts()
df["drive-wheels"].value_counts().to_frame()
drive_wheels_count = df["drive-wheels"].value_counts().to_frame()
drive_wheels_count.rename(columns={"drive-wheels":"drive_wheels_count"},inplace=True)
drive_wheels_count
drive_wheels_count.index.name = 'drive-wheels'
drive_wheels_count
engine_location_count = df["engine-location"].value_counts().to_frame()
engine_location_count.rename(columns={"engine-location":"engine-location-count"},inplace=True)
engine_location_count.index.name = 'engine-location'
engine_location_count
df["drive-wheels"].unique()
group_1 = df[["drive-wheels","body-style","price"]]
group_1 = group_1.groupby(["drive-wheels"],as_index=False).mean()
group_1
group_1 = df[["drive-wheels","body-style","price"]]
group_1 = group_1.groupby(["drive-wheels", "body-style"],as_index=False).mean()
group_1
df_pivot = group_1.pivot(index="drive-wheels",columns="body-style")
df_pivot
df_pivot = df_pivot.fillna(0)
df_pivot
group_2 = df[["body-style","price"]]
group_2 = group_2.groupby("body-style",as_index=False).mean()
group_2
plt.pcolor(df_pivot,cmap='RdBu')
plt.colorbar()
plt.show()
fig, ax = plt.subplots()
im = ax.pcolor(df_pivot, cmap='RdBu')
row_labels = df_pivot.columns.levels[1]
col_labels = df_pivot.index
ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is:",pearson_coef,",", "with a p-value of:",p_value)
pearson_coef,p_value=stats.pearsonr(df["horsepower"],df["price"])
pearson_coef,p_value
pearson_coef,p_value=stats.pearsonr(df["length"],df["price"])
pearson_coef,p_value
grouped_test2=group_1[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)
grouped_test2.get_group('4wd')['price']
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val) 
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val) 
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val) 
path = "cleaned_and_binned.csv"
df = pd.read_csv(path)
df.head()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm
X = df[["highway-mpg"]]
Y = df["price"]
lm.fit(X,Y)
Yhat = lm.predict (X)
Yhat[0:5]
lm.intercept_
lm.coef_
lm1 = LinearRegression()
lm1
lm1.fit(df[["engine-size"]],df[["price"]])
lm1
lm.intercept_
lm.coef_
Z = df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y = df["price"]
lm.fit(Z,Y)
lm.intercept_
lm.coef_
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses',"highway-mpg"]],df["price"])
lm2
lm2.coef_
lm2.intercept_
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
df[["peak-rpm","highway-mpg","price"]].corr()
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
Y_hat = lm.predict(Z)
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()
x = df["highway-mpg"]
y = df["price"]
f = np.polyfit(x, y, 3) # trying to fit a 3rd degree polynomial
p = np.poly1d(f)
p
PlotPolly(p,x,y,'highway-mpg')
np.polyfit(x,y,3)
f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
p1
PlotPolly(p1,x,y,"highway-mpg-11th")
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(input)
pipe.fit(Z,y)
yhat = pipe.predict(Z)
yhat[0:10]
X = df[["highway-mpg"]]
Y = df["price"]
lm.fit(X,Y)
lm.score(X,Y)
Yhat = lm.predict(X)
Yhat[0:5]
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
mse
lm.fit(Z,df["price"])
lm.score(Z,df["price"])
Yhat = lm.predict(Z)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
mse
from sklearn.metrics import r2_score   # Since we're using a different function, we'll need this
r_squared = r2_score(y, p(x))
r_squared
mean_squared_error(df['price'], p(x))
df=df._get_numeric_data()
df.head()
%%capture
! pip install ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
























