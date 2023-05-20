import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, re, pickle
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

da = pd.read_csv("hotel-classification-dataset.csv")
print(da.head())

# 1- preprocessing
da['Review_Date'] = pd.to_datetime(da['Review_Date'])
da['Review_Date'] = np.array(da['Review_Date'].dt.strftime("%m%d%Y"))

# drop not useful data and missing values
da.dropna(how='any', inplace=True)
da.drop_duplicates(inplace=True)
da.drop(['lat', 'lng'], axis=1)

# Handle outliers
q1 = da['Average_Score'].quantile(0.25)
q3 = da['Average_Score'].quantile(0.75)
iqr = q3 - q1
L_b = q1 - 1.5*iqr
U_b = q3 + 1.5*iqr
da = da[(da['Average_Score'] >= L_b) & (da['Average_Score'] <= U_b)]

def Feature_Encoder(X , cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
cols = ('Hotel_Address', 'Hotel_Name', 'Reviewer_Nationality', 'Negative_Review',
        'Positive_Review', 'Tags', 'days_since_review', 'Reviewer_Score')
Feature_Encoder(da, cols)

X = da.iloc[:, :-1]
Y = da['Reviewer_Score']
print("x.shape is : {}".format(X.shape))
print("y.shape is : {}".format(Y.shape))

print(da.describe())

#Get the correlation between the features
hotel_data = da.iloc[:, :]
corr = hotel_data.corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

'''top_feature = corr.index[abs(corr['Reviewer_Score']) > 0.2]
top_feature = top_feature.delete(-1)
X = X[top_feature]'''

print(da['Hotel_Name'].value_counts())
print("=" * 80)
print(da['Reviewer_Nationality'].value_counts())
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, shuffle=True , random_state=42)

# feature scaling(Normalization)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

def class_model(model):
    model.fit(X_train, y_train)
    model_name = str(type(model)).split('.')[-1][:-2]
    model_name = re.sub(r'\([^)]*\)', '', model_name)
    print("Train Error:[{}] = {} {}Test Error:[{}] = {}".format(model_name,
            metrics.mean_squared_error(y_train, model.predict(X_train)), ' , ', model_name,
            metrics.mean_squared_error(y_test, model.predict(X_test))))
    print("Train accuracy:[{}] = {}{}".format(model_name, metrics.accuracy_score(y_train, model.predict(X_train)) * 100 , '%'), ' , ',
          "Test accuracy:[{}] = {}{}".format(model_name, metrics.accuracy_score(y_test, model.predict(X_test)) * 100 , '%'))
    print("=" * 80)
    pass

logistic_class = class_model(LogisticRegression())
tree_class = class_model(DecisionTreeClassifier())
random_forest_class = class_model(RandomForestClassifier())
#svc = class_model(SVC(kernel = 'linear', gamma='scale', C=1))
Kneighbor_class = class_model(KNeighborsClassifier(n_neighbors=5))
GaussianNB_class = class_model(GaussianNB())
MLPClassifier = class_model(MLPClassifier(max_iter=100))
print(logistic_class, '\n', tree_class, '\n', random_forest_class, '\n',
      Kneighbor_class, '\n', GaussianNB_class, '\n', MLPClassifier)

# TEST SCRIPT
filename = 'logistic.pickle'
pickle.dump(MLPClassifier, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
new_data = ...
predictions = loaded_model.predict()
print("predictions: ", predictions)
