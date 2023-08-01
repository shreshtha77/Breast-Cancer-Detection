#Importing essential libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Loading the dataset
cancer = datasets.load_breast_cancer()


#Data analysis
# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


#Split the data into training sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) 


#Generating model
clf = svm.SVC(kernel='linear') # Linear Kernel

#Training the model
clf.fit(X_train, y_train)


#Predict the response for test dataset
y_pred = clf.predict(X_test)


