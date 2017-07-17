# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:56:12 2017

@author: Valued Customer
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import operator
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import ensemble

from sklearn.decomposition import PCA




def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
        
    """ bad fitting can produce negative predictions. In these cases we
        will just return a score of 10 and code will not crash """
    if np.min(y_predict) < 0.1:
        return 10
    
#    y_true_log = np.log(y_true)
#    y_pred_log = np.log(y_predict)
    
    from sklearn.metrics import mean_squared_error
    RMSE = (mean_squared_error(y_true, y_predict))**0.5
    
    # Return the score
    return RMSE

def anova(train, feature):
    """ This function takes the feature and forms a list of possible values
        in the dataframe train. It then composes a list of arrays for these
        values and does the analysis of variance on this"""
    values_list = train[feature]
    values_set = set(values_list)
    values_set = set(filter(lambda x: x == x , values_set))
    
    array_of_values = []
    for value in values_set:
        array_this_value = train['SalePrice'][train[feature] == value]
        array_of_values.append(array_this_value)
#        print (value)
    
    f_value, p_value = stats.f_oneway(*array_of_values)
    
    return f_value, p_value

def lin_check(train, feature):
    """ This function checks the correlation of the feature with the sale price"""
    x = train[feature][train[feature].notnull()]
    y = train['SalePrice'][train[feature].notnull()]
    
    x = x.as_matrix()
    y = y.as_matrix()
    x = np.squeeze(np.asarray(x))

    slope, intercept, r_value, pvalue, stderr = stats.linregress(x,y)
    
    yy = slope * x + intercept
    r2 = performance_metric(y, yy)
#    print (r, r2)
    
    return r_value

def evaluate_features(num_features, cat_features, train):
    """ This function goes through the features giving us an 
        evaluation of thier quality by doing a linear fit on the
        numerical ones and an analysis of variance on the catigorical
        ones """
        
    for feature in num_features:
        r = lin_check(train, feature)
        print ("r_value for ",feature, " = ",r)
        
    for feature in cat_features:
        f_value, p_value = anova(train, feature)
        print('f and p values for ', feature, '= {0:6.2f}  {1:7.2E}'.format(f_value,p_value))
        
    return
        

def label_encode(feature, train, test):
    """ This function picks on feature from a data frame then does labeling
        and encoding on it. The training and testing arrays must be labeled
        and encoded the same so this function must work on both sets simultaniously.
        It returns  matricies to be added to the X matrix for both train and test"""
    X_train_df = train[feature]
    X_test_df = test[feature]
    X_all_df = pd.concat([X_train_df, X_test_df], axis=0)
#    class_mapping = {label:idx for idx,label in enumerate(np.unique(train[feature]))}
    class_mapping = {label:idx for idx,label in enumerate(np.unique(X_all_df))}
    X_train_df = X_train_df.map(class_mapping)
    X_train = X_train_df.as_matrix()
    X_test_df = X_test_df.map(class_mapping)
    X_test = X_test_df.as_matrix()
    X_all_df = X_all_df.map(class_mapping)
    X_all = X_all_df.as_matrix()
    
    ohe = OneHotEncoder()
    ohe.fit(X_all.reshape(-1,1))
    X_train = ohe.transform(X_train.reshape(-1,1)).toarray()
    X_test = ohe.transform(X_test.reshape(-1,1)).toarray()
#    X_train = ohe.fit_transform(X_train.reshape(-1,1)).toarray()
#    X_test = X_train
    
    return X_train, X_test

def kfold_score(X, y, model):
    """ This function takes in an X matrix along with a y array and a model
        then does a kfold cross validation returning the average error and 
        standard deviation of the error. There are definitely better ways
        to do this with python 3.5 but this project is bound by 2.7 and this
        is the way I found that worked """
        
    rows = np.shape(X)[0]
    kf = KFold(rows,10)
    test_scores = []
    train_scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_score = performance_metric(y_test, y_pred_test)
        test_scores.append(test_score)
        y_pred_train = model.predict(X_train)
        train_score = performance_metric(y_train, y_pred_train)
        train_scores.append(train_score)
        
    test_scores_mean = np.mean(test_scores)
    test_scores_std = np.std(test_scores)
    train_scores_mean = np.mean(train_scores)
    train_scores_std = np.std(train_scores)
    
    return test_scores_mean, train_scores_mean, test_scores_std, train_scores_std

def find_best_feature(remain_num_features, remain_cat_features, best_X_orig, train, test):
    """ This function scans through the availiable features and returns the one 
        that gives the lowest score """
        
    best_score = 10.0
    
    for feature in remain_num_features:
#        print(feature)
        features = [feature]
        X = train.as_matrix(features)
        X = X.astype(float)
        imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imr = imr.fit(X)
        X = imr.transform(X)
        if np.size(best_X_orig) > 0:
            X = np.hstack((X, best_X_orig))
        y = np.log(train['SalePrice'])
        scaler = preprocessing.StandardScaler().fit(X)
        X_scl = scaler.transform(X)
        reg = linear_model.Ridge (alpha = global_alpha)
        score, junk, junk, junk = kfold_score(X_scl, y, reg)
        if score < best_score:
            best_score = score
            best_feature = feature
            best_X = X
            
    for feature in remain_cat_features:
#        print(feature)
        X, junk = label_encode(feature, train, test)
        if np.size(best_X_orig) > 0:
            X = np.hstack((X, best_X_orig))
        y = np.log(train['SalePrice'])
        scaler = preprocessing.StandardScaler().fit(X)
        X_scl = scaler.transform(X)
        reg = linear_model.Ridge (alpha = global_alpha)
        score, junk, junk, junk = kfold_score(X_scl, y, reg)
        if score < best_score:
            best_score = score
            best_feature = feature
            best_X = X
            
    return best_feature, best_score, best_X

def compose_feature_lists(num_features, cat_features, train, test, N_features):
    """ This function does the work of composing the lists of best numerical
        features and best categorical features."""
        
    best_num_features = []
    best_cat_features = []
    remain_num_features = num_features
    remain_cat_features = cat_features
    best_X = []

    for i in range(N_features):
        best_feature, best_score, best_X = find_best_feature(remain_num_features, remain_cat_features, best_X, train, test)

        if best_feature in remain_num_features:
            remain_num_features.remove(best_feature)
            best_num_features.append(best_feature)
            print ("best feature is a numerical feature", best_feature, best_score)
    
        if best_feature in remain_cat_features:
            remain_cat_features.remove(best_feature)
            best_cat_features.append(best_feature)
            print ("best feature is a catigorical feature", best_feature, best_score)
            
    return best_num_features, best_cat_features, best_X

def num_feature_examine(num_features, train):
    for feature in num_features:
#    for i in range(2):
#        feature = num_features[0]
        xx = train[feature]
        yy = train['SalePrice']
        plt.clf()
        plt.title(feature, fontsize=20)
        plt.plot(xx,yy,"o")
        print (feature)
        plt.show()
        raw_input("Press Enter to continue...") 
        
def rsq_function(x_sub, x_alpha, index, y):
    """ Function to be minimized. Takes in array with values found in original
        data frame and x values in interior of array. It adds 0 to beginning
        of array and 1 to the end, maps the variable then does linear fit
        returning 1 - r^2 """
    x_map = np.insert(x_sub,0, 0)
    x_map = np.append(x_map, [1])
    my_map = pd.Series(x_map, index)
    x = x_alpha.map(my_map)
    slope, intercept, r_value, pvalue, stderr = stats.linregress(x,y)
    return (1.0 - r_value * r_value)
    
        
def optimize_map(train, feature, index, first_index, last_index):
    """ This function determines a map for taking ordinal values to numerical
        values. first_index is the index to map to 0 and last_index is the index
        to map to 1. In some cases there might be an index after last_index such
        as NA to allow to find its own value """
    y = y = np.log(train['SalePrice'])
    x_alpha = train[feature]
    
    x_sub = np.linspace(1.0/len(index), 1.0 - 1.0/len(index), len(index)-2)
    res = minimize(rsq_function, x_sub, args=(x_alpha, index, y), method='nelder-mead', 
                   options={'xtol': 1e-8, 'disp': True})
    
    x_map = res.x
    x_map = np.insert(x_map,0, 0)
    x_map = np.append(x_map, [1])
    
    return(x_map)
    
""" First inpor the data. train_NAL2 is a modification of train.csv because NA
    which means No Alley is confused with Not Applicable """
train = pd.read_csv("train_NAL2.csv")
"""xx = train['OverallCond']
yy = train['SalePrice']
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.plot(xx,yy,"o")
fig = plt.figure()
plt.plot(xx,yy,"o")
plt.title('Saleprice vs Living Area', fontsize=20)
plt.xlabel('Living Area (Sq feet)', fontsize=18)
plt.ylabel('Saleprice ($)', fontsize=16)
fig.savefig('outliers.png')"""
train = train.drop([1298]).reset_index(drop=True)
train = train.drop([523]).reset_index(drop=True)

test = pd.read_csv("test_NAL.csv")

# FEATURES -----------------------------------------------------------------------------------

""" Divide features into two catigories, catigorical and numerical (catigorical
    and ordinal are in the same group for this exercise). """

cat_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                'Condition2', 'BldgType', 'HouseStyle', 'ExterQual', 'OverallCond',
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'Foundation', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'Heating', 'CentralAir', 'BsmtFinType2', 'KitchenQual',
                'Electrical', 'GarageType', 'FireplaceQu', 'GarageFinish',
                'PavedDrive', 'GarageQual', 'GarageCond', 'PoolQC',
                'Fence' , 'MiscFeature', 'SaleType', 'SaleCondition', 'BsmtFinType1'
                ]
    
    
num_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                'PoolArea', 'MiscVal', 'YrSold', 'BsmtFinSF1',
                'OverallQual','HeatingQC', 'Functional'
                ]

""" evaluate_features gives an overall evaluation of each feature. It does not need to be 
    done on each run"""    
#evaluate_features(num_features, cat_features, train)
""" Introduce mapping on categorical features to see if reduced categories will help"""
"""my_map = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train['OverallQual'] = (train['OverallQual']).map(my_map)
test['OverallQual'] = (test['OverallQual']).map(my_map) """
index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
opt_map = optimize_map(train, 'OverallQual', index, 0, 10)
my_map = pd.Series(opt_map, index)
train['OverallQual'] = (train['OverallQual']).map(my_map)
test['OverallQual'] = (test['OverallQual']).map(my_map)

index=['Ex', 'Gd', 'TA', 'Fa', 'Po']
opt_map = optimize_map(train, 'HeatingQC', index, 0, 4)
my_map = pd.Series(opt_map, index)
train['HeatingQC'] = (train['HeatingQC']).map(my_map)
test['HeatingQC'] = (test['HeatingQC']).map(my_map)

my_map = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'])
train['Functional'] = (train['Functional']).map(my_map)
test['Functional'] = (test['Functional']).map(my_map)


""" compose_feature_lists chooses the single feature that best helps the data be 
    fitted. It then finds the best of the remaining features which matches the feature
    when added to the first feature. It continues until the specified number of features
    are determined. It returns a list of the best categorical features and numerical 
    features. In the future a training matrix can be determined straight from these
    lists of features """
global_alpha = 60.0
"""best_num_features, best_cat_features, best_X = compose_feature_lists(num_features, 
                                                                     cat_features, train, test, 35)
with open('best_num_features.txt', 'wb') as fp:
    pickle.dump(best_num_features, fp)
    
with open('best_cat_features.txt', 'wb') as fp:
    pickle.dump(best_cat_features, fp) """
    
with open ('best_num_features.txt', 'rb') as fp:
    best_num_features = pickle.load(fp)
    
with open ('best_cat_features.txt', 'rb') as fp:
    best_cat_features = pickle.load(fp)
    
"""for feature in num_features:
    feature_col = train[feature]
    num_nans = feature_col.isnull().sum()
    print (feature, num_nans)
    
for feature in cat_features:
    feature_col = train[feature]
    num_nans = feature_col.isnull().sum()
    print (feature, num_nans)"""
    
#num_feature_examine(num_features, train)
    
num_features = best_num_features
X_num = train.as_matrix(num_features)
X_num = X_num.astype(float)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(X_num)
X_num = imr.transform(X_num)
y = np.log(train['SalePrice'])


cat_features = best_cat_features

X_cat, junk = label_encode(cat_features[0], train, test)

iter_cat_features = iter(cat_features)
next(iter_cat_features)
for feature in iter_cat_features:
    X_feature, junk = label_encode(feature, train, test)
    X_cat = np.hstack((X_cat, X_feature))


X = np.hstack((X_num, X_cat))

scaler = preprocessing.StandardScaler().fit(X)
X_scl = scaler.transform(X)

"""pca = PCA(n_components=125)
pca.fit(X_scl)
X_scl = pca.transform(X_scl)"""
"""lda = LDA(n_components=100)
lda.fit(X_scl, y)
X_scl = lda.transform(X_scl) """

"""reg = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate = 0.05, max_depth=3, 
                                         max_features='sqrt',min_samples_leaf=15, 
                                         min_samples_split=10, loss = 'huber')
cv_score, train_score, junk, junk = kfold_score(X_scl, y, reg)"""


alpha_mat = []
ratio_mat = []
cv_mat = []
cv_std_mat = []
train_mat = []
train_std_mat = []
alpha = 0.1
alpha = 10
#ratio = 0.05
while alpha < 250.0:
#while ratio < .9:
    print(alpha)
#    reg = linear_model.ElasticNet(alpha = alpha, l1_ratio=ratio)
#    reg = SVR(C=1.0/alpha, kernel = 'linear')
#    reg = SVR(kernel = 'rbf', epsilon = .005, C=1.0/alpha)
    reg = linear_model.Ridge (alpha = alpha)
    cv_score, train_score, cv_std, train_std = kfold_score(X_scl, y, reg)
    alpha_mat.append(alpha)
#    ratio_mat.append(ratio)
    cv_mat.append(cv_score)
    cv_std_mat.append(cv_std)
    train_mat.append(train_score)
    train_std_mat.append(train_std)
    alpha = alpha * 1.3
#    ratio = ratio + 0.05
plt.figure(1)
fig1 = plt.figure(1)
#plt.plot(alpha_mat, cv_mat, alpha_mat, train_mat)

cv_sum = map(operator.add, cv_mat, cv_std_mat)
cv_diff = map(operator.sub, cv_mat, cv_std_mat)
plt.plot(alpha_mat, cv_mat, color='blue',marker='o',markersize=5,label='cross validation error')
plt.fill_between(alpha_mat, cv_sum, cv_diff, alpha=0.15, color='blue')
train_sum = map(operator.add, train_mat, train_std_mat)
train_diff = map(operator.sub, train_mat, train_std_mat)
plt.plot(alpha_mat, train_mat, color='green',marker='o',markersize=5,label='training error')
plt.fill_between(alpha_mat, train_sum, train_diff, alpha=0.15, color='green')
plt.xscale('log')
plt.xlabel("Alpha (Regularization parameter)")
#plt.xlabel("l1_ratio")
plt.ylabel("Error")
plt.legend(loc = 'lower right')
plt.show()

fig1.savefig("valid.png")

X_scl_orig = X_scl
size = 110
reg = linear_model.Ridge (alpha = global_alpha)
while size < np.shape(X_scl_orig)[1]:
    pca = PCA(n_components=size)
    pca.fit(X_scl_orig)
    X_scl = pca.transform(X_scl_orig)
    cv_score, train_score, cv_std, train_std = kfold_score(X_scl, y, reg)
    print (size, cv_score)
    size = size + 2
    

"""epsilon_mat = []
cv_mat = []
train_mat = []
epsilon = 0.005
while epsilon < 0.2:
    print(epsilon)
#    reg = linear_model.Ridge (alpha = alpha)
    reg = SVR(kernel = 'rbf', epsilon = epsilon, C=1.0/1.0)
    cv_score, train_score, junk, junk = kfold_score(X_scl, y, reg)
    epsilon_mat.append(epsilon)
    cv_mat.append(cv_score)
    train_mat.append(train_score)
    epsilon = epsilon * 1.1
plt.figure(1)
plt.plot(epsilon_mat, cv_mat, epsilon_mat, train_mat)
plt.xscale('log')"""


reg = linear_model.Ridge(alpha = global_alpha)
#reg = SVR(kernel = 'linear', C=1.0/global_alpha)

# Train the model using the training sets
""" Do learning curves by training on various numbers of points to see if sample size is sufficient"""
points_mat = []
cv2_mat = []
cv2_std_mat = []
train2_mat = []
train2_std_mat = []
points = np.size(y)
train_points = 100
while train_points < points:
    rows_keep = range(train_points)
    X_sub = X_scl[rows_keep]
    cv_score, train_score, cv_std, train_std = kfold_score(X_sub, y, reg)
    points_mat.append(train_points)
    cv2_mat.append(cv_score)
    cv2_std_mat.append(cv_std)
    train2_mat.append(train_score)
    train2_std_mat.append(train_std)
    train_points = train_points + 50
plt.figure(2)   
fig2 = plt.figure(1) 
cv2_sum = map(operator.add, cv2_mat, cv2_std_mat)
cv2_diff = map(operator.sub, cv2_mat, cv2_std_mat)
plt.plot(points_mat, cv2_mat, color='blue',marker='o',markersize=5,label='cross validation error')
plt.fill_between(points_mat, cv2_sum, cv2_diff, alpha=0.15, color='blue')
train2_sum = map(operator.add, train2_mat, train2_std_mat)
train2_diff = map(operator.sub, train2_mat, train2_std_mat)
plt.plot(points_mat, train2_mat, color='green',marker='o',markersize=5,label='training error')
plt.fill_between(points_mat, train2_sum, train2_diff, alpha=0.15, color='green')
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.legend(loc = 'upper right')
plt.show()
fig2.savefig('learning.png')


"""fig = plt.figure(2)
plt.plot(points_mat, cv2_mat, points_mat, train2_mat)
fig.savefig('test.png') """

""" With parameters tested, construct and fit the final model """
X_scl = X_scl_orig
reg = linear_model.Ridge (alpha = global_alpha)
cv_score, train_score, junk, junk = kfold_score(X_scl, y, reg)
print ("best guess for kaggle is ", cv_score)
reg.fit(X_scl, y)

""" Construct the X matrix for the testing data """
num_features = best_num_features
X_num = test.as_matrix(num_features)
X_num = X_num.astype(float)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(X_num)
X_num = imr.transform(X_num)


cat_features = best_cat_features

junk, X_cat = label_encode(cat_features[0], train, test)

iter_cat_features = iter(cat_features)
next(iter_cat_features)
for feature in iter_cat_features:
    junk, X_feature = label_encode(feature, train, test)
    X_cat = np.hstack((X_cat, X_feature))
    
X_test = np.hstack((X_num, X_cat))

X_test_scl = scaler.transform(X_test)

"""X_test_scl_orig = X_test_scl
pca = PCA(n_components=130)
pca.fit(X_scl_orig)
X_scl = pca.transform(X_scl_orig)
X_test_scl = pca.transform(X_test_scl_orig)
reg = linear_model.Ridge (alpha = global_alpha)
reg.fit(X_scl, y)"""
reg = linear_model.Ridge (alpha = global_alpha)
reg.fit(X_scl_orig, y)
y_pred = reg.predict(X_test_scl)
y_pred = np.exp(y_pred)

ids_test = test['Id'].values

submission = pd.DataFrame({
            "Id": ids_test,
            "SalePrice": y_pred
            })
submission.to_csv("Iowa_housing.csv", index=False)




#import numpy.ma as ma
#np.where(np.isnan(X), ma.array(X, mask=np.isnan(X)).mean(axis=0), X)





#score = performance_metric(X, y)