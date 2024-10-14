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
#Function 3 Run the trained data across various models: 

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

def evaluate_models(X_train, y_train, seed=1234, scoring='accuracy'):
    """
    This function evaluates several machine learning models using k-fold cross-validation 
    and compares their performance. Additionally, it plots ROC curves for each model.
    
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
    X_train (DataFrame or ndarray): The training data (features).
    y_train (Series or ndarray): The training labels (target variable).
    seed (int): The random seed for reproducibility (default: 1234).
    scoring (str): The scoring metric used for evaluating model performance (default: 'accuracy').
    
    Returns:
    results (list): Cross-validation scores for each model.
    names (list): The names of the models.
    
    Example usage:
    results, names = evaluate_models(X_train, y_train)
    """
    
    # Define a dictionary of models to evaluate
    models = {
        "Logreg": LogisticRegression(solver='lbfgs', max_iter=1000),
        "NN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "NB": GaussianNB(),
    }
    
    # Initialize variables to store results
    results = []
    names = []
    
    # Perform k-fold cross-validation for each model
    for name, model in models.items():
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
        # Print the cross-validation results
        msg = f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})"
        print(msg)
        
        # ROC Curve generation
        if hasattr(model, "predict_proba"):
            y_probas = cross_val_predict(model, X_train, y_train, cv=kfold, method="predict_proba")
            y_scores = y_probas[:, 1]  # Use the probability of the positive class
            fpr, tpr, thresholds = roc_curve(y_train, y_scores)
            plot_roc_curve(fpr, tpr, name)
    
    # Boxplot to compare model performance
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    return results, names

# Example usage:
# results, names = evaluate_models(X_train, y_train)
