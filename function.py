###########################################################################
###########################################################################
####                                                                   ####
####           Functions that make life easier for DS work             #### 
####                                                                   ####
###########################################################################
###########################################################################


import pandas as pd

###########################################################################
# Function 0.1: Drop multiple columns as part of a flow aka can cont if error 
def Drop_1ormore_col(dataframe, *column_names):
    """
    This function removes one or more specified columns from a DataFrame.
    
    Parameters:
    dataframe (DataFrame): The DataFrame from which columns will be removed.
    *column_names: Variable length argument list of column names to drop (separated by commas).
    
    Returns:
    The DataFrame is updated in-place, and a message is printed indicating how many columns were removed.
    
    Example usage:
    Drop_1ormore_col(df, 'Age', 'Fare', 'Cabin')
    """
    
    cnt = 0  # Counter to track number of columns removed
    
    # Iterate through the provided column names
    for col in column_names:
        if col in dataframe.columns:
            dataframe.drop(col, axis=1, inplace=True)
            cnt += 1
        else:
            print(f'The column "{col}" is not in the dataframe')
    
    print(f'\nTotal {cnt} columns removed\n...process continue...\n')

# Example usage:
# Drop_1ormore_col(df, 'Age', 'Fare', 'Cabin')

###########################################################################
# Function 0.2: Drop multiple columns as part of a flow aka can cont if error 
def Solid_nums(df, *vars):
    """
    This function checks if specified columns exist in the DataFrame and converts their content 
    to numeric values. If the conversion is not possible, it will handle the error gracefully.
    
    Parameters:
    df (DataFrame): The DataFrame that contains the data.
    *vars: Variable length argument list of column names to be converted to numeric.
    
    Returns:
    The DataFrame is updated in-place. The process will continue, and if a column does not exist 
    or cannot be converted, a message will be printed.
    
    Example usage:
    Solid_nums(df, 'age', 'income', 'score')
    """
    for var in vars:
        if var in df.columns:
            try:
                # Attempt to convert the column to numeric, forcing errors to NaN where necessary
                df[var] = pd.to_numeric(df[var], errors='coerce')
                print(f'The column "{var}" has been converted to numeric.')
            except Exception as e:
                print(f'Error converting column "{var}" to numeric: {e}')
        else:
            print(f'The column "{var}" is not in the dataframe.')
    
    print('Process continues...\n')

# Example usage:
# Solid_nums(df, 'age', 'income', 'score')


###########################################################################
# Function 1.1: Union dataset with reporting stats. 
def stack_2_datasets(data1 , data1name , data2 , data2name):
    """
    This function is designed to append data from two sources and return a single combined dataset.
    This stack_2_datasets function takes two datasets and vertically combines (stacks) them, while adding an additional column calls 'file' to each dataset to identify the source of the data. 
        It also prints the shape (number of rows and columns) of each individual dataset and the combined dataset.
    Parameter: 
        data1: The first dataset (usually a pandas DataFrame) that will be stacked.
        data1name: A string representing the name or identifier for the first dataset. This will be added as a value in a new 'file' column.
        data2: The second dataset (usually a pandas DataFrame) that will be stacked.
        data2name: A string representing the name or identifier for the second dataset. This will be added as a value in a new 'file' column.
    #Example
        #fulltableau = stack_2_datasets(testfull,'test',train,'train')
    """
    data1['file'] = data1name
    data2['file'] = data2name
    full = pd.concat([data1, data2], ignore_index=True, sort=False)
    print(data1name, 'has', data1.shape ,'rows/columns')
    print(data2name, 'has', data2.shape ,'rows/columns')
    print('After combining vertically, the new data now has', full.shape,'rows/columns')
    return full 

###########################################################################
# Function 1.2: Seperate Datasets:
def XYsplit(data, targetvar):
    """
    This function is designed to split a dataset into target and predictor variables.
    It separates the target variable (the column you want to predict) from all other columns (predictors).
    
    Parameters:
    data (DataFrame): The dataset to be split.
    targetvar (str): The name of the target variable column.
    
    Returns:
    y_train (Series): The target variable.
    x_train (DataFrame): The predictor variables (all columns except the target variable).
    
    # Example
    # y_train, x_train = XYsplit(train, 'Survived')
    """
    y_train = data[targetvar]
    x_train = data.drop(columns=[targetvar])
    print("Sample target variable (y_train):\n", y_train.sample(1))
    print("Sample predictor variables (x_train):\n", x_train.sample(5))
    return y_train, x_train

###########################################################################
# Function 2: Quick and dirty data transformation pipelines:

from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific columns from a pandas DataFrame.
    
    Parameters:
    attribute_names (list): List of column names to select.
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

# Custom transformer for imputing the most frequent value
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that fills missing values in categorical columns with the most frequent value.

    Require Packages:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputerer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import FeatureUnion
    """
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

#Examples: 
# Pipeline for numeric columns
#num_pipeline = Pipeline([
#    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),  # Select numeric columns
#    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with the median
#])
#
## Pipeline for categorical columns
#cat_pipeline = Pipeline([
#    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),  # Select categorical columns
#    ("imputer", MostFrequentImputer()),  # Impute missing values with the most frequent value
#    ("cat_encoder", OneHotEncoder(sparse=False)),  # One-hot encode categorical variables
#])
#
## Combine the numeric and categorical pipelines
#preprocess_pipeline = FeatureUnion(transformer_list=[
#    ("num_pipeline", num_pipeline),
#    ("cat_pipeline", cat_pipeline),
#])
#
## Apply the pipeline to the training dataset
#X_train = preprocess_pipeline.fit_transform(x_train)
#
## View the preprocessed data
#X_train

###########################################################################
#Function 3.1 Plot the roc curve of a model - more of a support
def plot_roc_curve(fpr, tpr, model_name):
    """
    Plots the ROC curve for a given model.

    Require Packages: 
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_curve, cross_val_predict
    
    Parameters:
    fpr (array): False Positive Rate values for the ROC curve.
    tpr (array): True Positive Rate values for the ROC curve.
    model_name (str): Name of the model being evaluated (used as the title).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line representing random chance
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.show()

###########################################################################
#Function 3.2 Support function to find the best threshold of a model (the best point on a RoC curve graph): 
def find_best_threshold(y_true, y_scores):
    """
    Find the best threshold based on F1 score for binary classification.

    Required Packages:
    from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    
    Parameters:
    - y_true: Ground truth binary labels.
    - y_scores: Predicted scores or probabilities for the positive class.
    
    Returns:
    - best_threshold: The threshold that maximizes the F1 score.
    - best_f1: The F1 score at the best threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

###########################################################################
#Function 3.3 The big model to connect the section 3 and run the models evaluation: 
def evaluate_models(models, X_train, y_train, scoring='accuracy', n_splits=10):
    """
    This function evaluates multiple models using cross-validation and compares their performance
    through boxplots. It also calculates ROC-AUC for models with probability predictions and suggests
    the best model based on the highest accuracy and AUC score.
    
    Required Packages:
    from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn import model_selection

    Parameters:
    - models (dict): A dictionary where keys are model names and values are instantiated model objects.
    - X_train (DataFrame or array): The training input data.
    - y_train (Series or array): The target variable for training.
    - scoring (str): The evaluation metric (default is 'accuracy').
    - n_splits (int): Number of folds for cross-validation (default is 10).
    
    Returns:
    - models (dict): The input models.
    - results (list of arrays): Cross-validation results for each model.
    - names (list): The names of the models.
    - best_model_name (str): The name of the best model based on accuracy or AUC.
    - best_thresholds (dict): The best threshold for each model (if applicable).

    Example:
    from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

    models = {
        "Logreg": LogisticRegression(solver='lbfgs', max_iter=1000),
        "NN": KNeighborsClassifier(),
        #"LinearSVM": SVC(probability=True, kernel='linear'), #class_weight='balanced'
        "GBC": GradientBoostingClassifier(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "NB": GaussianNB(),
    }

    # Run the eval
    models, results, names, best_acc_model, best_auc_model, best_thresholds = evaluate_models(models, X_train, y_train, scoring='accuracy', n_splits=5)
    """
    
    results = []
    names = []
    mean_scores = []
    auc_scores = {}
    best_thresholds = {}  # To store the best threshold for each model
    best_model_acc_threshold = None  # To store the best threshold for the accuracy model
    best_auc_model_acc = None  # To store the accuracy for the best AUC model
    
    for name, model in models.items():
        kfold = KFold(n_splits=n_splits)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        mean_score = cv_results.mean()
        mean_scores.append(mean_score)
        msg = f"{name}: {mean_score:.4f} ({cv_results.std():.4f})"
        print(msg)

        # Generate ROC curve and calculate AUC for models with predict_proba
        if hasattr(model, "predict_proba"):
            y_probas = cross_val_predict(model, X_train, y_train, cv=kfold, method="predict_proba")
            y_scores = y_probas[:, 1]  # Use the probability of the positive class
            fpr, tpr, thresholds = roc_curve(y_train, y_scores)
            
            auc = roc_auc_score(y_train, y_scores)
            auc_scores[name] = auc
            print(f"{name}: AUC = {auc:.4f}")
            plot_roc_curve2(fpr, tpr, name)
            
            # Find the best threshold based on F1 score
            best_threshold, best_f1 = find_best_threshold(y_train, y_scores)
            best_thresholds[name] = best_threshold
            print(f"{name}: Best threshold = {best_threshold:.4f}, Best F1 score = {best_f1:.4f}")

            # Evaluate model performance at the best threshold
            y_pred_at_best_threshold = (y_scores >= best_threshold).astype(int)
            acc = accuracy_score(y_train, y_pred_at_best_threshold)
            precision = precision_score(y_train, y_pred_at_best_threshold)
            recall = recall_score(y_train, y_pred_at_best_threshold)
            f1 = f1_score(y_train, y_pred_at_best_threshold)
            print(f"{name} performance at best threshold: Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

            # Track the best threshold for the accuracy model
            if name == names[np.argmax(mean_scores)]:
                best_model_acc_threshold = best_threshold    
                
            # Track the accuracy for the best AUC model
            if auc == max(auc_scores.values()):
                best_auc_model_acc = acc
            
    # Boxplot algorithm comparison
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.ylabel(scoring.capitalize())
    plt.show()

    # Suggest the best model based on accuracy
    best_acc_index = np.argmax(mean_scores)
    best_model_acc_name = names[best_acc_index]
    print(f"\nThe best model based on accuracy is: {best_model_acc_name} with an average score of {mean_scores[best_acc_index]:.4f}")
    print(f"The best threshold for {best_model_acc_name} is: {best_model_acc_threshold:.4f}")
    
    # Suggest the best model based on AUC (if applicable)
    if auc_scores:
        best_auc_model = max(auc_scores, key=auc_scores.get)
        print(f"The best model based on AUC is: {best_auc_model} with an AUC score of {auc_scores[best_auc_model]:.4f} and an accuracy of {best_auc_model_acc:.4f}")
    else:
        best_auc_model = None

    return models, results, names, best_model_acc_name, best_auc_model, best_thresholds


def export_model(models,best_model_acc_name):
    """
    Export the selected model as a pickle file - a support function

    Required Package:
        pickle
    """
    best_model_object = models[best_model_acc_name]  # Save the model object for exporting
    with open(f"{best_model_acc_name}_model.pkl", 'wb') as f:
        pickle.dump(best_model_object, f)
    print(f"\nModel ({best_model_acc_name}) has been saved as '{best_model_acc_name}_model.pkl'.")