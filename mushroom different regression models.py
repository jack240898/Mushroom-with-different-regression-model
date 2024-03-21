import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # train_test
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier,RidgeClassifier,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import LinearSVC, SVC,NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))
    
##DATA DESCRIPTION
import warnings
warnings.filterwarnings(action='ignore')
##("E:\M.E. ECE\sem1\Machine learning\ml\mushrooms.csv")
df = pd.read_csv("E:\M.E. ECE\sem1\Machine learning\ml\mushrooms.csv")

# Change the names of the class to be more explicit
# with "edible" and "poisonous"
df['class'] = df['class'].map({"e": "edible", "p": "poisonous"})
df.iloc[:5,:8]

## DATA VISULIZATION
df['class'].value_counts().plot.bar(figsize = (8,5), color = ['grey','red'])
plt.xticks(rotation=0)
plt.title('Quantity of each class in the dataset', fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
## Data Preprocessing
from sklearn.preprocessing import LabelEncoder

X = df.drop("class", axis = 1).copy()
y = df['class'].copy()

label_encoder_data = X.copy()
label_encoder = LabelEncoder()
for col in X.columns:
    label_encoder_data[col] = label_encoder.fit_transform(label_encoder_data[col])
    X = label_encoder_data
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

## model prediction
# Create a dictionary with the model which will be tested
models = {
    "KNeighborsClassifier":{"model":KNeighborsClassifier() },
    "DecisionTreeClassifier":{"model":DecisionTreeClassifier() },
    "ExtraTreeClassifier":{"model":ExtraTreeClassifier() },
    "LogisticRegression":{"model":LogisticRegression() },
    "SVC":{"model":SVC() },
    "RandomForestClassifier":{"model":RandomForestClassifier() },
}
# Use the 10-fold cross validation for each model
# to get the mean validation accuracy and the mean training time
for name, m in models.items():
    # Cross validation of the model
    model = m['model']
    result = cross_validate(model, X_train,y_train,cv = 10)
    
    # Mean accuracy and mean training time
    mean_val_accuracy = round( sum(result['test_score']) / len(result['test_score']), 4)
    mean_fit_time = round( sum(result['fit_time']) / len(result['fit_time']), 4)
    
    # Add the result to the dictionary witht he models
    m['val_accuracy'] = mean_val_accuracy
    m['Training time (sec)'] = mean_fit_time
    
    # Display the result
    print(f"{name:27} mean accuracy using 10-fold cross validation: {mean_val_accuracy*100:.2f}% - mean training time {mean_fit_time} sec")
    
    # Create a DataFrame with the results
models_result = []

for name, v in models.items():
    lst = [name, v['val_accuracy'],v['Training time (sec)']]
    models_result.append(lst)

df_results = pd.DataFrame(models_result, 
                          columns = ['model','val_accuracy','Training time (sec)'])
df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
df_results.reset_index(inplace=True,drop=True)
df_results

### Mean Validation Accuracy for each model
plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'val_accuracy', data = df_results)
plt.title('Mean Validation Accuracy for each Model\ny-axis between 0.8 and 1.0', fontsize = 15)
plt.ylim(0.8,1.005)
plt.xlabel('Model', fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.xticks(rotation=90, fontsize=12)
plt.show()

## Training time is Set
plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'Training time (sec)', data = df_results)
plt.title('Training time for each Model in sec', fontsize = 15)
plt.xticks(rotation=90, fontsize=12)
plt.xlabel('Model', fontsize=15)
plt.ylabel('Training time (sec)',fontsize=15)
plt.show()

### Prediction metrics of the best model using the test set
# Get the model with the highest mean validation accuracy
best_model = df_results.iloc[0]

# Fit the model
model = models[best_model[0]]['model']
model.fit(X_train,y_train)

# Predict the labels with the data set
pred = model.predict(X_test)

# Display the results
printmd(f'## Best Model: {best_model[0]} with {best_model[1]*100}% accuracy on the test set')
printmd(f'## Trained in: {best_model[2]} sec')

# Display a confusion matrix
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize = (10,7))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()