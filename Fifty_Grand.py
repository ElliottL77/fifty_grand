import matplotlib.axis
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_curve, precision_recall_curve
from numpy import sqrt, argmax
pd.set_option('display.max_columns', 10)

os.chdir("D:\Documents\Stat_Projects\Classify_Income")
print('CuRrEnT WD {0}'.format(os.getcwd()))
#
fields = ["age", "workclass", "fnlwgt","education", "education-num", "marital-status", "occupation",
          "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
          "yearly-income"]  # sets it up to import these variables.

fifty_csv = 'income_train.csv'
fifty_test_csv = "income_test.csv"
fifty = pd.read_csv(fifty_csv, skipinitialspace=True,
                     usecols=fields)  # Imports and uses only the columns we defined above.
fifty_test = pd.read_csv(fifty_test_csv, skipinitialspace=True,
                     usecols=fields)  # Imports and uses only the columns we defined above.
fifty = pd.DataFrame(fifty)
fifty_test = pd.DataFrame(fifty_test)
fifty.head()  # Gives us the first six rows (to include column names).
# fifty_test.head()
# fifty.columns  # Lists columns.

# ####### Exploring the data. ########

# round(fifty['yearly-income'].value_counts() / float(len(fifty['yearly-income'])), 2)  # Shows the percentage of 0's to 1's.
# round(fifty_test['yearly-income'].value_counts() / float(len(fifty_test['yearly-income'])), 2)  # Shows the percentage of 0's to 1's.

## 0.76(0s) to 0.24(1s) in both cases.

# fifty.info()
# fifty.isnull().sum()  # checks how many variables have null values.
# fifty_test.isnull().sum()  # checks how many variables have null values.


# round(fifty.describe(), 2)  # Gives mean, std, quartiles of  non-catigorical variables.
cat_vars = np.array(
    ["workclass","education", "marital-status", "occupation",
          "relationship", "race", "sex", "native-country",
          "yearly-income"])  # array of cat vars for future use.
#
# for i in range(len(cat_vars)):
#     print(fifty[cat_vars[i]].value_counts())  # Gives list of categories and count of each.

######## Changing variable type. #######

fifty[cat_vars] = fifty[cat_vars].astype("category")  # changes our array variables into categorical variable Dtypes.
fifty["age"] = fifty["age"].astype('int64')  # Age should just be an integer.

fifty_test[cat_vars] = fifty_test[cat_vars].astype("category")  # changes our array variables into categorical variable Dtypes.
fifty_test["age"] = fifty_test["age"].astype('int64')  # Age should just be an integer.

fifty.info()  # The conversion was successful.
fifty_test.info()

####### Changing values. #######
fifty.replace("?", -99, inplace=True) # Replaces all "?" cases with -99 value.
fifty_test.replace("?", -99, inplace=True) # Replaces all "?" cases with -99 value.

# fifty.isnull().sum() # Sums the number of NULL values in each column.

## workclass, occupation and native-country have missing/"?" values.

# for i in range(len(cat_vars)):
#     print(fifty[cat_vars[i]].value_counts())  # Gives list of categories and count of each.

## the category and classes to collapse them to.
## education - <HS, HS, Ass/tech, Bach, Masters, Doc.
## marital-status - Married, Never-married, Divorced/Seperated, Widowed.
## Occupation - ADDRESS LATER
## relationship - consider dropping.
## Race - White, black, other
## native-country - US, NA, SA, Europe (EU), Asia (AS) * drop "south".
### Setting up the mapping dictionary ###

cleanup_nums = {"sex": {'Male': 0, 'Female': 1},
                "race": {'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo':2, 'Other':2},
                "education": {"Preschool": 1, "1st-4th":1, "5th-6th": 1, "7th-8th":1, "9th":1, "10th":1,
                              "11th":1, "12th":1, "HS-grad":0, "Some-college":2, "Assoc-acdm":3, "Assoc-voc":3, "Bachelors":4, "Prof-school":5, "Masters":6, "Doctorate":7},
                "marital-status": {"Married-civ-spouse":0, "Married-spouse-absent":0, "Married-AF-spouse":0, "Never-married":1, "Divorced":2, "Separated":2, "Widowed":3},
                "native-country": {'United-States':0, 'Canada':1, 'Puerto-Rico':2, 'Honduras':2, 'Jamaica':2, 'Guatemala':2, 'Nicaragua':2, 'El-Salvador':2, 'Trinadad&Tobago':2, 'Peru':2,
                                   'Mexico':2, 'Dominican-Republic':2, 'Cuba':2, 'Ecuador':2, 'Haiti':2, 'Columbia':2, 'England':3, 'Germany':3, 'Greece':3, 'Italy':3, 'Poland':3, 'Ireland':3,
                                   'Portugal':3, 'France':3, 'Hungary':3,  'Scotland':3, 'Yugoslavia':3, 'Holand-Netherlands':3, 'Cambodia':4, 'Outlying-US(Guam-USVI-etc)':4, 'India':4,
                                   'Japan':4, 'South':4, 'China':4, 'Iran':4, 'Philippines':4, 'Vietnam':4, 'Laos':4, 'Taiwan':4, 'Thailand':4, 'Hong':4,},
                "workclass": {'Private':0, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':1, 'Local-gov':1, 'State-gov':1, 'Without-pay':2, 'Never-worked':2},
                "yearly-income": {'<=50K':0, '<=50K.':0, '>50K':1, '>50K.':1}
                }

fifty = fifty.replace(cleanup_nums)
fifty_test = fifty_test.replace(cleanup_nums)

####### Dropping columns. #######
fifty_weight = fifty['fnlwgt']
fifty_test_weight = fifty_test['fnlwgt']
fifty.drop(['fnlwgt','education-num', 'occupation', 'relationship'], axis=1, inplace=True)
fifty_test.drop(['fnlwgt','education-num', 'occupation', 'relationship'], axis=1, inplace=True)
# fifty.head()
fifty.info()

####### Exploratory plots. #######

color_things = ['red', 'green', 'blue']
red_color = dict(color=color_things[0])
green_color = dict(color=color_things[1])
blue_color = dict(color=color_things[2])
age_fig = fifty.boxplot(column="age", boxprops=red_color, medianprops=green_color,
                        whiskerprops=red_color, capprops=red_color,
                        flierprops=dict(markeredgecolor=color_things[2])
                        )
age_fig.set_title('B&W plot of Age')
age_fig.set_ylabel('Age')

# Age has high end outliers.

hour_fig = fifty.boxplot(column="hours-per-week", boxprops=red_color, medianprops=green_color,
                         whiskerprops=green_color, capprops=red_color, flierprops=dict(markeredgecolor=color_things[2])
                         )
hour_fig.set_title('B&W plot of Hours Per Week')
hour_fig.set_ylabel('Hours Per Week')

# HPW shows lots of outliers on both sides.

age_histo = fifty['age'].hist(bins=20, color='#ca1120')
age_histo.set_title('Histogram_of_Age')
age_histo.set_xlabel('age')
age_histo.set_ylabel('Count')
# Heavy right-skew.

# Shows a peak at the mean and extremes.

hours_histo = fifty['hours-per-week'].hist(bins=20, color='#0000c1')
hours_histo.set_title('Histogram_of_HPW')
hours_histo.set_xlabel('Hours Per Week')
hours_histo.set_ylabel('Count')
# Massive spike around 40, distributed all over the place otherwise.

cg_histo = fifty['capital-gain'].hist(bins=20, color='green')
cg_histo.set_title('Histogram_of_capital-gain')
cg_histo.set_xlabel('capital-gain')
cg_histo.set_ylabel('Count')

cl_histo = fifty['capital-loss'].hist(bins=20)
cl_histo.set_title('Histogram_of_capital-loss')
cl_histo.set_xlabel('capital-loss')
cl_histo.set_ylabel('Count')

sns.set_theme(style="darkgrid")
sexbar = sns.countplot(x="sex",data=fifty, palette=['#ff0066', '#0044ff'], alpha=0.75)
plt.show()

edbar = sns.countplot(x="education",data=fifty, order=[0,1,2,3,4,5,6,7], alpha=0.75)
plt.show()

regionbar = sns.countplot(x="native-country",data=fifty, alpha=0.75, order=[0,1,2,3,4,-99])
plt.show()

####### Testing for Multicoliniarity, distributions, independence, etc. #######
corr_coeff = fifty.corr()
print(corr_coeff)

## All have corr_coeffs around 0.

# cross1 = pd.crosstab(fifty['marital-status'], fifty["native-country"], margins=True)  # X^2 test.
# cross1_stats = stats.chi2_contingency(cross1) [1]
# print(cross1_stats)

# ind_cat_vars = np.array(
#     ["workclass","education", "marital-status", "race", "sex", "native-country"
#      ])
# lst = []
#
# for i in range(len(ind_cat_vars)):
#     for j in range(len(ind_cat_vars)):
#         chi2 = pd.crosstab(fifty[ind_cat_vars[i]], fifty[ind_cat_vars[j]], margins=True)
#         chi2_stats = stats.chi2_contingency(chi2)[1]
#         lst.append(chi2_stats)
# print(lst)

# Running VIF check.
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

Y,X = dmatrices('fifty["yearly-income"]~ age + workclass + education + fifty["marital-status"] + '
                'race + sex + fifty["capital-gain"] + fifty["capital-loss"] + fifty["hours-per-week"] + fifty["native-country"]', data=fifty, return_type = 'dataframe'
                )
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['variable'] = X.columns
print(vif)

## Native-country T=2 (Latin America) has highest VIF at 5.42. It was here I realized native-country and race, while both having low VIFs, seems redundant to the other.
## I'll remove race as it's not as wide-spread as native-country.

fifty.drop(['race'], axis=1, inplace=True)
fifty_test.drop(['race'], axis=1, inplace=True)

####### Pipelining #######
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

Y = fifty['yearly-income']
fifty.drop(['yearly-income'], axis=1, inplace=True)
X = fifty

Y2 = fifty_test['yearly-income']
fifty_test.drop(['yearly-income'], axis =1, inplace=True)
X2 = fifty_test

cont_cols = np.array(['age', 'hours-per-week', 'capital-gain', 'capital-loss'])
cat_cols = np.array(['workclass', 'education', 'marital-status', 'sex', 'native-country','yearly-income'])

cat_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown='ignore'))
    ]
)
num_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",StandardScaler())
    ]
)

cont_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

cont_cols2 = X2.select_dtypes(include="number").columns
cat_cols2 = X2.select_dtypes(exclude="number").columns

process = ColumnTransformer(
    transformers=[
        ("numeric", num_pipe, cont_cols),
        ("categorical", cat_pipe, cat_cols)
    ]
)

X_process = process.fit_transform(X)
Y_process = SimpleImputer(strategy="most_frequent").fit_transform(
    Y.values.reshape(-1,1)
)


X_process2 = process.fit_transform(X2)
Y_process2 = SimpleImputer(strategy="most_frequent").fit_transform(
    Y2.values.reshape(-1,1)
)

##Getting column names and changing them.
pipe_names = process.get_feature_names_out()
# age, capital-gain, capital-loss, hours-per-week, workclass(-99,0,1,2), education(0,1,2,3,4...7),
# marital_status(0-3), sex(0,1), native_country(-99,0,1,2,3,4)

# name_list = [
#  "Age", "Capital_Gain", "Capital_Loss", "Hours-Per-Week", "Workclass(missing)", "Workclass(Private or Self-Emp.)", "Workclass(Public)",
#  "Workclass(No_Pay or Never_Worked)", "Education(HS-grad)", "Education(<HS)", "Education(Some_college)", "Education(AD)",
#  "Education(BD)", "Education(Pro_School)", "Education(MD)", "Education(PhD)", "Marital_Status(Married)", "Marital_Status(Never_married)", "Marital_Status(Divorced)",
#  "Marital_Status(Widowed)", "Sex(Male)", "Sex(Female)", "Native_Country(Missing)", "Native_Country(USA)", "Native_Country(Canada)",
#  "Native_Country(Central/South_America)", "Native_Country(Europe)", "Native_Country(Asia)"
# ]
#
# X_process = pd.DataFrame(X_process)
# X_process2 = pd.DataFrame(X_process2)
# Y_process = pd.DataFrame(Y_process)
# Y_process2 = pd.DataFrame(Y_process2)
#
# X_process.columns = [
#     "Age","CapGain","CapLoss","Hours","Workclass(missing)","Work(Priv)",
#     "Workclass(Public)","Workclass(UnEmp)","Education(HSGrad)","Education(LessHS)","Education(SomeCol)",
#     "Education(AD)","Education(BD)","Education(ProSch)","Education(MD)","Education(PhD)","Marital_Status(Married)",
#     "Marital_Status(NeverMarried)","Marital_Status(Divorced)","Marital_Status(Widowed)","Sex(Male)","Sex(Female)","Native_Country(Missing)","Native_Country(USA)","Native_Country(Canada)",
#     "Native_Country(CentSouAmerica)","Native_Country(Europe)","Native_Country(Asia)"
#     ]
# X_process2.columns = [
#     "Age","CapGain","CapLoss","Hours","Workclass(missing)","Work(Priv)",
#     "Workclass(Public)","Workclass(UnEmp)","Education(HSGrad)","Education(LessHS)","Education(SomeCol)",
#     "Education(AD)","Education(BD)","Education(ProSch)","Education(MD)","Education(PhD)","Marital_Status(Married)",
#     "Marital_Status(NeverMarried)","Marital_Status(Divorced)","Marital_Status(Widowed)","Sex(Male)","Sex(Female)","Native_Country(Missing)","Native_Country(USA)","Native_Country(Canada)",
#     "Native_Country(CentSouAmerica)","Native_Country(Europe)","Native_Country(Asia)"
#     ]
# Y_process.columns = ["50K"]
# Y_process2.colum = ["50K"]
#######

# Pipeline.get_feature_names_out(cat_cols, cont_cols)

####### "Splitting" the data into train/test #######
X_train = X_process
Y_train = Y_process

X_test = X_process2
Y_test = Y_process2

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold

# X_train, X_test, Y_train, Y_test = train_test_split(X_process, Y_process, test_size=0.3,
#                                                     random_state=1111)  # Splits data into train/test sections. Random_state = seed.
#
# ####### SMOTE #######
# from imblearn.over_sampling import SMOTENC
# from imblearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
#
# sm = SMOTENC(categorical_features=[1,2], random_state=1111)
# X_train, Y_train = sm.fit_resample(X_train, Y_train)
# X_sm_te, Y_sm_te = sm.fit_resample(X_test, Y_test)
#
# round(Y_train.value_counts() / float(len(Y_train)), 2)  # Shows the percentage of 0's to 1's.
#
# pipe = Pipeline(steps = [('smotenc', SMOTENC(categorical_features=[0,2,3], random_state = 1111)),
#                       ('standardscaler', StandardScaler()),
#                       ('logisticregression', LogisticRegression())])
# pipe.fit(X_train, Y_train)# cross validation using intra-fold sampling
# cross_validate(pipe, X_train, Y_train)
#
#
# ####### Balanced Random Forest Classifier #######
# from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.datasets import make_classification
#
# # X_train, Y_train = make_classification()
# clf = BalancedRandomForestClassifier( random_state=1111)
# clf.fit(X_train,Y_train)

### Determining optimal Hyperparameter. ###
# from sklearn.model_selection import GridSearchCV
#
# parameters = [
#     {'C':[1,10,100,1000], 'kernel':['linear']},
#     {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
#     {'C':[1,10,100,1000], 'kernel':['poly'], 'degree':[2,3,4] , 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05]}
# ]
#
# grid_search = GridSearchCV(estimator=svc,
#                            param_grid=parameters,
#                            scoring = 'accuracy',
#                            cv = 5,
#                            verbose=0)
#
# grid_search.fit(X_resam, Y_resam)
# #
# print('GridSearchCV best score : {:.4f}\n\n'.format(grid_search.best_score_))
# print('Parameters that give the best results :', '\n\n', (grid_search.best_params_))
# print('\n\nEstimator chosen :','\n\n', (grid_search.best_estimator_))
#
# # C = 1000, kernel = 'rbf', gamma = 0.9.
#
# print('GridSearchCV score on test set: {0:0.4f}'.format(grid_search.score(X_test, Y_test)))
# # 0.7645.

####### Running the SVM #######
# from sklearn.svm import SVC
#
# svc = SVC(class_weight='balanced', probability=True)
#
# svc.fit(X_train,Y_train) # Fits classifier to training set.
# # svc.fit(X_resam, Y_resam) #Uses our SMOTE sets.
#
# Y_pred=svc.predict(X_test) # Makes predictions on test set.
#
# print('Accuracy w/ default hyperparameters: {0:0.4f}'. format(accuracy_score(Y_test,Y_pred))) #0.9517.
# confusion_fu = confusion_matrix(Y_test,Y_pred) # Prints the whole confusion matrix.
# print(classification_report(Y_test, Y_pred))
#
# ### The SVM is not giving great results. Will use Random Forest
#
####### Running the Gradient Boosting Forest ######
# from sklearn.ensemble import GradientBoostingClassifier
#
# gbc = GradientBoostingClassifier(learning_rate=0.9)
# gbc.fit(X_sm, Y_sm)
# Y_pred_gbc = gbc.predict(X_test)
#
# print(classification_report(Y_test, Y_pred_gbc))
# confusion_fu_gbc = confusion_matrix(Y_test, Y_pred_gbc)

####### Running Random Forest #######
# from sklearn.ensemble import RandomForestClassifier
#
# rf = RandomForestClassifier(n_estimators=1000, max_features=2, random_state=1111)
# rf.fit(X_train, Y_train)
# Y_pred_rf = rf.predict(X_test)
#
# print(classification_report(Y_test, Y_pred_rf))
# print(confusion_matrix(Y_test, Y_pred_rf))

####### KNN #######
# from sklearn.neighbors import KNeighborsClassifier
# cl = KNeighborsClassifier(n_neighbors=5)
# cl.fit(X_sm, Y_sm)
#
# Y_pred_knn = cl.predict(X_test)
#
# print(confusion_matrix(Y_test,Y_pred_knn))
# print(classification_report(Y_test, Y_pred_knn))

####### ADAboost #######
# from sklearn.ensemble import AdaBoostClassifier
#
# ada=AdaBoostClassifier(n_estimators=1000, learning_rate=1.9992, random_state=1111)
# ada.fit(X_train, Y_train)
# Y_pred_ada = ada.predict(X_test)
#
# print(classification_report(Y_test, Y_pred_ada))
# print(confusion_matrix(Y_test, Y_pred_ada))

####### XGBoost, try it. #######
import xgboost as xgb

boostah = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, max_depth=10,
                            learning_rate=0.01, n_jobs=-1, max_delta_step=10, scale_pos_weight= 3, base_score=np.mean(Y_train),
                            eval_metric="logloss"
                            ) # scale_pos_weight is a weight. #0s / #1s . 0.00001
boostah.fit(X_train,Y_train)
predict = boostah.predict(X_test)

prob_preds = boostah.predict_proba(X_test)
prob_preds = prob_preds[:,1] #Keeps only the positive outcomes.

fpr, tpr, thresholds = roc_curve(Y_test, prob_preds)

pyplot.plot([0,1], [0,1], linestyle="-")
pyplot.plot(fpr, tpr)
pyplot.show()

gmeans = sqrt(tpr *(1-fpr))
ix = argmax(gmeans)
print ('Best threshold=%f, G-Mean%.3f' % (thresholds[ix], gmeans[ix])) #0.538021

precision, recall, thresholds = precision_recall_curve(Y_test, prob_preds)
no_skill = len(Y_test[Y_test == 1]) / len(Y_test)
pyplot.plot([0,1], [no_skill, no_skill], linestyle = '--')
pyplot.plot(recall, precision, marker=".")
pyplot.show()

fscore = (2*precision * recall) / (precision + recall)
iy = argmax(fscore)
print("Best threshold=%f, fscore=%.3f" % (thresholds[iy], fscore[iy])) #0.513975

nu_threshold = np.where(prob_preds >= 0.69,1,0)

# print('Accuracy = ', accuracy_score(Y_test, predict))
# print("F1 Score = ", f1_score(Y_test, predict))
# print(classification_report(Y_test, predict))
# print(confusion_matrix(Y_test, predict))

print('Accuracy = ', accuracy_score(Y_test, nu_threshold)) # 0.76
print("F1 Score = ", f1_score(Y_test, nu_threshold)) # 0.63
print(classification_report(Y_test, nu_threshold))
print(confusion_matrix(Y_test, nu_threshold))

# 10974 1461
# 1437 2409

xgb.plot_importance(boostah)
# Number of times a variable occurs in the trees of the model.
plt.title("xgboost.plot_importance(boostah)")
plt.show()

xgb.plot_importance(boostah, importance_type="cover")
#Relative obs related to variable.
plt.title("xgboost.plot('cover')")
plt.show()

xgb.plot_importance(boostah,importance_type="gain")
#Takes each variable's contribution for each tree. Higher values means more important.
# Gain is usually the one you want to take the most stock in interpreting results. (https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7)
plt.title("...('gain')")
plt.show()

### List of variables given they all return as "fX"

#  f0 - "Age"
#  f1 - "Capital_Gain"
#  f2 - "Capital_Loss"
#  f3 - "Hours-Per-Week"
#  f4 - "Workclass(missing)"
#  f5 - "Workclass(Private or Self-Emp.)"
#  f6 - "Workclass(Public)"
#  f7 - "Workclass(No_Pay or Never_Worked)"
#  f8 - "Education(HS-grad)"
#  f9 - "Education(<HS)"
#  f10 - "Education(Some_college)"
#  f11 - "Education(AD)"
#  f12 - "Education(BD)"
#  f13 - "Education(Pro_School)"
#  f14 - "Education(MD)"
#  f15 - "Education(PhD)"
#  f16 - "Marital_Status(Married)"
#  f17 - "Marital_Status(Never_married)"
#  f18 - "Marital_Status(Divorced)"
#  f19 - "Marital_Status(Widowed)"
#  f20 - "Sex(Male)"
#  f21 - "Sex(Female)"
#  f22 - "Native_Country(Missing)"
#  f23 - "Native_Country(USA)"
#  f24 - "Native_Country(Canada)"
#  f25 - "Native_Country(Central/South_America)"
#  f27 - "Native_Country(Europe)"
#  f28 - "Native_Country(Asia)"

# Var Importance plot, top five vars: Age, Hours, CapGain, CapLoss, Sex(Male).
# Cover plot's top five vars: WorkClass(Unemployed), Education(PhD), Native_Country(Canada), Education(Pro_School), CapGain
# Gain plot's top five vars: Marital_Stats(Married), Education(<HS), CapGain, Ed(HS-grad), Ed(Some_College)