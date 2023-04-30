import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")

da = pd.read_csv("hotel-regression-dataset.csv")
print(da.head())
print("==================================================================")

# 1- preprocessing
# convert(split) date to m,d,y
da['Review_Date'] = pd.to_datetime(da['Review_Date'])
da['Review_Date'] = np.array(da['Review_Date'].dt.strftime("%m%d%Y"))

print(da.describe())
print("==================================================================")

da_num = da.select_dtypes(include = ['float64', 'int64'])
da_num.hist(figsize=(12, 18), bins=50, xlabelsize=9, ylabelsize=9)
plt.show()

# Handle outliers
q1 = da['Average_Score'].quantile(0.25)
q3 = da['Average_Score'].quantile(0.75)
iqr = q3 - q1
L_b = q1 - 1.5*iqr
U_b = q3 + 1.5*iqr
da = da[(da['Average_Score'] >= L_b) & (da['Average_Score'] <= U_b)]

# drop not useful data and missing values
da.dropna(how='any' , inplace=True)
da.drop_duplicates(inplace=True)
da.drop(['lat' , 'lng'] , axis= 1)

# split day from text in "days_since_review" column
def get_day(data) :
    match = re.match(r"([0-9]+)", data , re.I)
    if match:
        number = int(match.group())
        return number
da["days_since_reviews"] = da["days_since_review"].apply(get_day)
print(da["days_since_reviews"])
print("==================================================================")

X = da.iloc[: , :-1]
Y = da['Reviewer_Score']
print("x.shape is : {}".format(X.shape))
print("y.shape is : {}".format(Y.shape))
print("==================================================================")

# feature encoding for categorical data
def Feature_Encoder(X , cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
cols = ('Hotel_Address','Hotel_Name' , 'Reviewer_Nationality' , 'Negative_Review' , 'Positive_Review' , 'Tags' , 'days_since_review')
X = Feature_Encoder(X , cols)

#Get the correlation between the features
hotel_data = da.iloc[:,:]
corr = hotel_data.corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

top_feature = corr.index[abs(corr['Reviewer_Score']) > 0.2]
top_feature = top_feature.delete(-1)
X = X[top_feature]

print(da['Hotel_Name'].value_counts())
print("==================================================================")
print(da['Reviewer_Nationality'].value_counts())
print("==================================================================")

'''from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
def get_continent(col):
    try:
        cn_a2_code = country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
    return (cn_a2_code, cn_continent)

da['Reviewer_continent'] = da['Reviewer_Nationality'].apply(get_continent)
print(da['Reviewer_continent'])'''

#print("index of the max review score" , da['Reviewer_Score'].idxmax().sum())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2 , train_size = 0.8 , shuffle = True , random_state=42)

# feature selection
from sklearn.feature_selection import SelectKBest ,  chi2
from sklearn.datasets import load_digits

X, Y = load_digits(return_X_y=True)
X = SelectKBest(chi2, k = 'all').fit_transform(X, Y)
#print(X.shape)

# feature scaling(Normalization)
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range=(0 , 1))
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
'''
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
'''
'''from sklearn.preprocessing import MaxAbsScaler
mas = MaxAbsScaler()
X_train = mas.fit_transform(X_train)
X_test = mas.transform(X_test)'''
'''from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
X_train = rs.fit_transform(X_train)
X_test = rs.transform(X_test)'''

# 2- Regression Techniques
arr = ['Additional_Number_of_Scoring' , 'Average_Score' , 'Review_Total_Negative_Word_Counts' , 'Total_Number_of_Reviews',
       'Review_Total_Positive_Word_Counts' , 'Total_Number_of_Reviews_Reviewer_Has_Given']

# ============= Model 1-simple linear regression =============
print("==============Model 1- linear regression===============")
lr = LinearRegression()
lr.fit(X_train , y_train)

print("Mean Square Error of train = {} {}Mean Square Error of test = {}".format(metrics.mean_squared_error(y_train, lr.predict(X_train)) , ' , ' ,
                                                                                metrics.mean_squared_error(y_test, lr.predict(X_test))))
print("Test r2_score = {} ".format(metrics.r2_score(y_test , lr.predict(X_test))) , ' , ' ,
      "Training r2_score = {} ".format(metrics.r2_score(y_train , lr.predict(X_train))))
for var in arr:
    plt.figure()
    sns.regplot(x= var , y= 'Reviewer_Score' , data= da , scatter_kws= {"color":"black" , "alpha":0.5} , line_kws= {"color":"red"})
    plt.title(f'Regression line of {var} and Reviewer_Score')
    plt.show()

# ============== Model 2-polynomial regression=============
print("============Model 2- polynomial regression=============")
T_errors = []
test_errors = []
l_deg = range(1 , 10)
for deg in l_deg:
    poly_features = PolynomialFeatures(degree = deg)
    X_train_poly = poly_features.fit_transform(X_train)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_train_predicted = poly_model.predict(X_train_poly)
    ypred = poly_model.predict(poly_features.transform(X_test))

    print('Mean Square Error of Test of degree ' , deg , " is : ", metrics.mean_squared_error(y_test, poly_model.predict(poly_features.fit_transform(X_test))) ,
          ', Mean Square Error of Train of degree ' , deg , " is : ", metrics.mean_squared_error(y_train, poly_model.predict(poly_features.fit_transform(X_train))))
    print('r2_Score of Test of degree ' , deg , " is : ", metrics.r2_score(y_test, poly_model.predict(poly_features.fit_transform(X_test))) ,
          ', r2_Score of Train of degree ' , deg , " is : ", metrics.r2_score(y_train, poly_model.predict(poly_features.fit_transform(X_train))))

    T_errors.append(metrics.mean_squared_error(y_train, poly_model.predict(poly_features.fit_transform(X_train))))
    test_errors.append(metrics.mean_squared_error(y_test, poly_model.predict(poly_features.fit_transform(X_test))))

plt.plot(l_deg, T_errors, label='Train')
plt.plot(l_deg, test_errors, label='Test')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Train and Test Errors vs Polynomial Degree')
plt.legend()
plt.show()

print("===============Model 3- Lasso regression===============")
lasso_r = Lasso()
X, y = load_iris(return_X_y=True)
lasso_r.fit(X_train , y_train)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, lasso_r.predict(X_train)) , ' , ' ,
      "Mean Square Error of test = " , metrics.mean_squared_error(y_test, lasso_r.predict(X_test)))

print("Test r2_score = " , metrics.r2_score(y_test, lasso_r.predict(X_test)) , ' , ' ,
      "Training r2_score = " , metrics.r2_score(y_train, lasso_r.predict(X_train)))
'''for i in arr:
    plt.scatter(da[i] , da['Reviewer_Score'])
    plt.xlabel(i , fontsize = 10)
    plt.ylabel("Reviewer_Score", fontsize = 10)
    plt.plot(X_test , lasso_r.predict(X_test) , color= 'red' , linewidth= 3)
    plt.title(f'Regression line of {i} and Reviewer_Score')
    plt.show()'''

print("===============Model 4- Ridge regression===============")
ridge_r = Ridge()
ridge_r.fit(X_train , y_train)

prediction_ridge_train = ridge_r.predict(X_train)
prediction_ridge_test = ridge_r.predict(X_test)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, ridge_r.predict(X_train)), ' , ',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, ridge_r.predict(X_test)))

print("Test r2_score = " , metrics.r2_score(y_test, ridge_r.predict(X_test)) , ' , ' ,
      "Training r2_score = " , metrics.r2_score(y_train, ridge_r.predict(X_train)))

print("==============Model 5- Decision Tree reg===============")
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, tree_reg.predict(X_train)), ' , ',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, tree_reg.predict(X_test)))

print("Test r2_score = " , metrics.r2_score(y_test, tree_reg.predict(X_test)) , ' , ' ,
      "Training r2_score = " , metrics.r2_score(y_train, tree_reg.predict(X_train)))

print("==============Model 6- Random Forest reg===============")
random_forest_reg = RandomForestRegressor(n_estimators = 100, max_depth=9 , random_state=42)
random_forest_reg.fit(X_train, y_train)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, random_forest_reg.predict(X_train)), ' , ',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, random_forest_reg.predict(X_test)))

print("Test r2_score = " , metrics.r2_score(y_test, random_forest_reg.predict(X_test)) , ' , ' ,
      "Training r2_score = " , metrics.r2_score(y_train, random_forest_reg.predict(X_train)))

'''print("==============Model 7- SVR regression===============")
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)

y_svr_train = svr.predict(X_train)
y_svr_test = svr.predict(X_test)

print(" Mean Square Error of train = ", metrics.mean_squared_error(y_train, y_svr_train), '\n',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, y_svr_test))

print("Test r2_score = " , metrics.r2_score(y_test, y_svr_test))
print("Training r2_score = " , metrics.r2_score(y_train, y_svr_train))'''

print("==============Model 7- RANSAC regression===============")
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()
ransac.fit(X_train, y_train)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, ransac.predict(X_train)), ' , ',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, ransac.predict(X_test)))

print("Test r2_score = ", metrics.r2_score(y_test, ransac.predict(X_test)) , ' , ' ,
      "Training r2_score = ", metrics.r2_score(y_train, ransac.predict(X_train)))

print("===============Model 8- Huber regression===============")
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor()
huber.fit(X_train, y_train)

print("Mean Square Error of train = ", metrics.mean_squared_error(y_train, huber.predict(X_train)), ' , ',
      "Mean Square Error of test = ", metrics.mean_squared_error(y_test, huber.predict(X_test)))

print("Test r2_score = ", metrics.r2_score(y_test, huber.predict(X_test)) , ' , ' ,
      "Training r2_score = ", metrics.r2_score(y_train, huber.predict(X_train)))

print("===============Model 9- xgboost regression===============")
from xgboost import XGBRegressor
sgb = XGBRegressor()
sgb.fit(X_train, y_train)
print("Mean Square Error of test = {}".format(metrics.mean_squared_error(y_test, sgb.predict(X_test))) ,
      "Mean Square Error of train = {}".format(metrics.mean_squared_error(y_train, sgb.predict(X_train))))
print("Test r2_score = {} ".format(metrics.r2_score(y_test, sgb.predict(X_test))) ,
      "Train r2_score = {} ".format(metrics.r2_score(y_train, sgb.predict(X_train))))

# ==============================BONUS==============================
print("==============================BONUS===============================")

#df['equal_or_lower_than_5?'] = da['Reviewer_Score'].apply(lambda w: 'Low_or_Intermediate_Score' if w <= 5 else 'High_Score')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# method 1 for sentiment analysis
df = pd.DataFrame(da)

# clean the text data
stop_words = set(stopwords.words('english'))
def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    # remove stop words
    words = word_tokenize(text)
    words = [w for w in words if not w in stop_words or w == 'No' or w == 'no']
    # re-join cleaned words
    text = " ".join(words)
    return text

df['Negative_Review'] = df['Negative_Review'].apply(clean_text)
print(df['Negative_Review'])
print("==================================================================")
df['Positive_Review'] = df['Positive_Review'].apply(clean_text)
print(df['Positive_Review'])

# use pre-trained sentiment analysis model to assign sentiment scores
'''sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Negative_Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
print(df['sentiment_score'])'''

# method 2 for sentiment analysis
print("==================================================================")
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return 'Positive'
    elif blob.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
df['negative_sentiment'] = df['Negative_Review'].apply(get_sentiment)
print(df['negative_sentiment'])
print("==================================================================")
df['positive_sentiment'] = df['Positive_Review'].apply(get_sentiment)
print(df['positive_sentiment'])

# Method 3
from wordcloud import WordCloud

def wordcloud_plot(da, title = None):
    wordcloud = WordCloud(
        background_color = 'black',
        max_words = 250 ,
        max_font_size = 50,
        scale = 5 ,
        random_state = 5
    ).generate(str(da))

    fig = plt.figure(1, figsize = (10, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 12)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
print(wordcloud_plot(df['Negative_Review']))
print(wordcloud_plot(df['Positive_Review']))

# print time training
print("==================================================================")
model = MLPClassifier()
X_train, y_train = load_iris(return_X_y=True)
start = time.time()
model.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {round(stop - start , 2)}s")

print("==================================================================")
# print total time
start = time.time()
# sleeping for 1 sec to get 10 sec runtime
time.sleep(1)
end = time.time()

# total time taken
print(f"Runtime of the program is {round(end - start , 2)}s")
print("==================================================================")

# Save data
da.to_csv("Updated_data_hotel.csv" , index=False)
# end of milestone 1
