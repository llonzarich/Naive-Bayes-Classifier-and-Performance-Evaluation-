import numpy as np # use numpy's random number generation
# from mysklearn import myevaluation
# from mysklearn.myevaluation import train_test_split, kfold_split, bootstrap_sample, accuracy_score
from mysklearn import myclassifiers
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier
from mysklearn import myevaluation
from mysklearn.myevaluation import binary_precision_score, binary_recall_score, binary_f1_score



def my_discretizer(y):
    '''
        Purpose: maps numeric predictions to strings: "high" or "low".

        Arguments: 
            y (float): a continuous variable (that we'll turn into a string category: "high" or "low").
    '''
    if y >= 100:
        return "high"
    else:
        return "low"
    

def mpg_discretizer(y):
    '''
        Purpose: maps continuous mpg predictions to a DOE category 1-10.
    
        Arguments: 
            y (float): a continuous variable (that we'll turn into categorical)
    '''
    # assign categorical categories (1-10) based on its continuous mpg value. 
    if y <=13:
        return 1
    elif 13 <= y <= 14:
        return 2
    elif 14 < y <= 16:
        return 3
    elif 16 < y <= 19:
        return 4
    elif 19 < y <= 23:
        return 5
    elif 23 < y <= 26:
        return 6
    elif 26 < y <= 30:
        return 7
    elif 30 < y <= 36:
        return 8
    elif 36 < y <= 44:
        return 9
    else:
        return 10
    

def random_subsample(X, y, k, classifier_class):
    '''
        Purpose: Call train_test_split() in a loop to generate distinct train/test splits, which will be used to evaluate model performance. 

        Arguments: 
            X (list of lists of obj's): - the list of samples
                                        - has shape: (n_samples, n_features)
            y (list of obj): - The target y values (labels corresponding to X)
                             - Default is None (in this case, the calling code only wants to sample X)
            k (int): the number of folds. aka, the number of times we'll generate train and test splits.
            classifier_class (class obj): - the classifier class we'll use to fit the model and predict.

        Returns:
            avg_acc (int): the avg accuracy of the fitted model over all k splits.
            avg_err_rate (int): the avg error rate of the fitted model over all k splits. 
    '''
    from mysklearn.myevaluation import train_test_split, accuracy_score

    # initialize lists to store the acc and error rate of the model on each of the 10 splits of the data. 
    accuracies = []
    err_rates = []

    # iterate k=10 times so we can evaluate model performance on 10 different subsets of data.
    for split in range(k):
        # create train and test sets by calling the train_test_split function in myevaluation.py.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        # create a classifier object (because we want a fresh classifier for each new split of data).
        classifier = classifier_class()

        # train the classifier on the training data (samples and corresponding labels). 
        classifier.fit(X_train, y_train)

        # predict MPG for the test instances.
        y_pred = classifier.predict(X_test)

        # convert continuous predicted mpg values to categorical ratings (aka discretize y into bins).
        pred_ratings = [mpg_discretizer(y) for y in y_pred]

        # convert the continuous true mpg values to a categorical rating using the discretizer.
        actual_ratings = [mpg_discretizer(y) for y in y_test]

        # compute the accuracy and error rate of the model by comparing true and predicted mpg.
        acc = accuracy_score(actual_ratings, pred_ratings)
        err = 1 - acc

        accuracies.append(acc)
        err_rates.append(err)

    # find the avg accuracy and avg error rate of the model across all 10 splits of data.  
    avg_acc = sum(accuracies) / k
    avg_err_rate = sum(err_rates) / k

    return avg_acc, avg_err_rate


def cross_val_predict(X, y, k, classifier_class, stratify=None):
    '''
        Purpose: - compute the k-fold cross-validaton for k = 10 and evaluate model performance for each split.
                 - aka, partition the data into 10 equal folds, and use 1 to be the test set for each iteration (NO repeated test sets).

        Arguments: 
            X (list of lists of obj's): - the list of samples
                                        - has shape: (n_samples, n_features)
            y (list of obj): - The target y values (labels corresponding to X)
                             - Default is None (in this case, the calling code only wants to sample X)
            k (int): the number of folds. aka, the number of times we'll generate train and test splits.
            classifier_class (class obj): - the classifier class we'll use to fit the model and predict.

        Returns:
            avg_acc (int): the avg accuracy of the fitted model over all k splits.
            avg_err_rate (int): the avg error rate of the fitted model over all k splits. 
            y_trues (list of strings): a list of all true mpg values in the dataset. 
            y_preds (list of strings): a list of all predicted mpg values.  
    '''
    from mysklearn.myevaluation import kfold_split, stratified_kfold_split, accuracy_score
    
    # initialize lists to store the acc, error rate, precision, recall, and f1-score of the model on each of the 10 splits of the data. 
    accuracies = []
    err_rates = []
    precisions = []
    recalls = []
    f1s = []

    # initalize lists to store all the predicted class labels and all the true class labels (for the confusion matrix)
    y_trues = []
    y_preds = []

    # split the dataset into cross-validation folds. 
    # note: kfold_split returns a list of tuples where each tuple has the train and test indicies for a given fold.
    if stratify == False:
        folds = kfold_split(X, n_splits=k, shuffle=True)
    if stratify == True:
        folds = stratified_kfold_split(X, y, n_splits=k, shuffle=True)

    # iterate over each train/test split so we can evaluate model performance on the different subsets of data.
    for train_indices, test_indices in folds:
        # convert X and y to numpy arrays ONLY for slicing
        X_array = np.array(X, dtype=object)
        y_array = np.array(y, dtype=object)

        # create train and test sets for the current fold.
        X_train, y_train = X_array[train_indices].tolist(), y_array[train_indices].tolist()
        X_test, y_test = X_array[test_indices].tolist(), y_array[test_indices].tolist()

        # create a classifier object (because we want a fresh classifier for each new split of data).
        classifier = classifier_class()

        # train the classifier on the training data (samples and corresponding labels). 
        classifier.fit(X_train, y_train)

        # predict MPG for the test instances.
        y_pred = classifier.predict(X_test)

        pred_ratings = y_pred
        actual_ratings = y_test

        # compute the accuracy and error rate of the model by comparing true and predicted mpg.
        acc = accuracy_score(actual_ratings, pred_ratings)
        err = 1 - acc
        precision = binary_precision_score(actual_ratings, pred_ratings, labels=None, pos_label=None)
        recall = binary_recall_score(actual_ratings, pred_ratings, labels=None, pos_label=None)
        f1 = binary_f1_score(actual_ratings, pred_ratings, labels=None, pos_label=None)

        accuracies.append(acc)
        err_rates.append(err)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        y_trues.extend(actual_ratings)
        y_preds.extend(pred_ratings)

    # find the avg accuracy and avg error rate of the model across all 10 splits of data.  
    avg_acc = sum(accuracies) / k
    avg_err_rate = sum(err_rates) / k
    avg_precision = sum(precisions) / k
    avg_recall = sum(recalls) / k
    avg_f1 = sum(f1s) / k

    # convert y_trues and y_pred sto strings
    y_trues = [str(y) for y in y_trues]
    y_preds = [str(y) for y in y_preds]

    return avg_acc, avg_err_rate, avg_precision, avg_recall, avg_f1, y_trues, y_preds


def bootstrap_method(X, y, k, classifier_class): 
    '''
        Purpose: Split dataset into bootstrapped training set and out of bag test set and evaluate model performance on each split.

        Arguments: 
            X (list of lists of obj's): - the list of samples
                                        - has shape: (n_samples, n_features)
            y (list of obj): - The target y values (labels corresponding to X)
                             - Default is None (in this case, the calling code only wants to sample X)
            k (int): the number of folds. aka, the number of times we'll generate train and test splits.
            classifier_class (class obj): - the classifier class we'll use to fit the model and predict.

        Returns:
            avg_acc (int): the avg accuracy of the fitted model over all k splits.
            avg_err_rate (int): the avg error rate of the fitted model over all k splits. 
    '''
    from mysklearn.myevaluation import bootstrap_sample, accuracy_score

    # initialize lists to store the acc and error rate of the model on each of the 10 splits of the data. 
    accuracies = []
    err_rates = []

    # iterate over each train/test split so we can evaluate model performance on the different subsets of data.
    for split in range(k):
        # create train and test sets by calling the train_test_split function in myevaluation.py.
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(X, y) 

        # create a classifier object (because we want a fresh classifier for each new split of data).
        classifier = classifier_class()

        # train the classifier on the training data (samples and corresponding labels). 
        classifier.fit(X_sample, y_sample)

        # predict MPG for the test instances.
        y_pred = classifier.predict(X_out_of_bag)

        # convert continuous predicted and true mpg values to categorical ratings using the discretizer.
        pred_ratings = [mpg_discretizer(y) for y in y_pred]
        actual_ratings = [mpg_discretizer(y) for y in y_out_of_bag]

        # compute the accuracy and error rate of the model by comparing true and predicted mpg.
        acc = accuracy_score(actual_ratings, pred_ratings)
        err = 1 - acc

        accuracies.append(acc)
        err_rates.append(err)

    # find the avg accuracy and avg error rate of the model across all 10 splits of data.  
    avg_acc = sum(accuracies) / k
    avg_err_rate = sum(err_rates) / k

    return avg_acc, avg_err_rate
