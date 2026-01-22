# pylint: skip-file
# import sys, os
# sys.path.append(os.path.abspath("."))
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from mysklearn.myutils import my_discretizer

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier, \
    MyNaiveBayesClassifier


def test_simple_linear_regression_classifier_fit():
    '''
        Purpose: test case for the fit() method of MySimpleLinearRegressionClassifier.
        - ensure the classifier functions correctly -- computes the slope and intercept correctly.
    '''
    # set random seed for reproducibility.
    np.random.seed(0)
    
    # create a small dataset.
    X_train = [[val] for val in range(0, 100)]
    y_train = [2 * x[0] + np.random.normal(0, 25) for x in X_train]

    # create classifier object and fit it (aka, compute slope and intercept vals).
    my_lin = MySimpleLinearRegressionClassifier(discretizer=my_discretizer)
    my_lin.fit(X_train, y_train)

    # assert against a known correct implementation (sklearn's LinearRegression).
    # (step 1) create the sklearn classifier object and fit it using our dataset.
    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    
    # (step 2) actually assert to see if our model and the sklearn model give the same slope and intercept vals.
    assert np.isclose(my_lin.slope, sk_lin_reg.coef_[0])
    assert np.isclose(my_lin.intercept, sk_lin_reg.intercept_)
    
    # assert that when the slope is 0, the program is also giving a slope of 0. 
    X_train = [[3], [6], [8], [9], [11]]
    y_train = [4, 4, 4, 4, 4]
    my_lin1 = MySimpleLinearRegressor()
    my_lin1.fit(X_train, y_train)
    assert np.isclose(my_lin1.slope, 0)


def test_simple_linear_regression_classifier_predict():
    '''
        Purpose: - test case for the predict() method of MySimpleLinearRegressionClassifier.
                 - ensure the classifier predicts correct output labels. 
        
        Note: - choosing to assert against desk calculations. 
    '''
    # set random seed for reproducibility.
    np.random.seed(0)


    # ===== test case 1 ===========
    # create a simple linear regression classifier object. 
    my_lin_1 = MySimpleLinearRegressionClassifier(discretizer=my_discretizer)
    
    # manually set known slope and intercept vals. 
    my_lin_1.slope = 1
    my_lin_1.intercept = 0

    # create test data. 
    X_test_1 = [[2], [3], [4]]

    # call predict() to generate label predictions for test data. 
    y_pred_labels_1 = my_lin_1.predict(X_test_1)

    # define the correct labels for test data points.
    expected_labels_1 = ["low", "low", "low"]

    assert np.array_equal(y_pred_labels_1, expected_labels_1)


    # ===== test case 2 ===========
    # create a classifier object using my model. 
    my_lin_2 = MySimpleLinearRegressionClassifier(discretizer=my_discretizer)
    
    # manually set known slope and intercept vals.
    my_lin_2.slope = 2
    my_lin_2.intercept = 10
    
    # create test data points.
    X_test_2 = [[50], [75], [250]]

    # call predict() to generate label predictions for test data points.
    y_pred_labels_2 = my_lin_2.predict(X_test_2)

    # define the correct labels for test data points.
    expected_labels_2 = ["high", "high", "high"]

    assert np.array_equal(y_pred_labels_2, expected_labels_2)


def test_kneighbors_classifier_kneighbors():
    '''
        Purpose: test for finding the number of neighbors that an unknown test instance has in the dataset.
    '''
    # set random seed for reproducibility.
    np.random.seed(0)
    
    # test case 1: use the 4-instance training set example, asserting against desk check
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    # create a kNN classifier object and train it.
    my_knn1 = MyKNeighborsClassifier(n_neighbors=2)
    my_knn1.fit(X_train_class_example1, y_train_class_example1)

    # create a test data point (that we want to assign a label to based on the labels of its k nearest neighbors).
    X_test1 = [[0.9,0]]

    dists1, indices1 = my_knn1.kneighbors(X_test1)

    expected_indices1 = [[1, 2]]

    assert np.array_equal(indices1, expected_indices1)


    # test case 2: use the 8-instance training set example, asserting against desk check.
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    
    # create a kNN classifier object and train it.
    my_knn2 = MyKNeighborsClassifier(n_neighbors=2)
    my_knn2.fit(X_train_class_example2, y_train_class_example2)

    # create a test data point (that we want to assign a label to based on the labels of its k nearest neighbors).
    X_test2 = [[8,8]]

    dists2, indices2 = my_knn2.kneighbors(X_test2)

    expected_indices2 = [[1, 3]]

    assert np.array_equal(indices2, expected_indices2)


    # test case 3: use Bramer 3.6 self assessment exercise 2, asserting against exercise solution in Bramer Appendix E. 
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    
    # create a kNN classifier object and train it.
    my_knn3 = MyKNeighborsClassifier(n_neighbors=2)
    my_knn3.fit(X_train_bramer_example, y_train_bramer_example)

    X_test3 = [[3, 10]]
    dists3, indices3 = my_knn3.kneighbors(X_test3)

    expected_indices3 = [[1, 2]]

    assert np.array_equal(indices3, expected_indices3)


def test_kneighbors_classifier_predict():
    '''
        Purpose: test for the predicted labels of the knn model.
    '''
    np.random.seed(0)
    
    # test case 1: use the 4-instance training set example, asserting against desk check
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    my_knn1 = MyKNeighborsClassifier(n_neighbors=3)
    my_knn1.fit(X_train_class_example1, y_train_class_example1)

    # create an unseen instance that we want to assign a label to.
    X_test1 = [[0,0]]

    # generate a prediction.
    y_pred1 = my_knn1.predict(X_test1)

    expected_y_pred1 = ["good"]

    assert np.array_equiv(y_pred1, expected_y_pred1)


    # test case 2: use the 8-instance training set example, asserting against desk check.
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    my_knn2 = MyKNeighborsClassifier(n_neighbors=3)
    my_knn2.fit(X_train_class_example2, y_train_class_example2)

    # create an unseen instance that we want to assign a label to.
    X_test2 = [[8, 8]]

    # generate a prediction.
    y_pred2 = my_knn2.predict(X_test2)

    expected_y_pred2 = ["yes"]

    assert np.array_equiv(y_pred2, expected_y_pred2)


    # test case 3: use Bramer 3.6 self assessment exercise 2, asserting against exercise solution in Bramer Appendix E. 
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    
    my_knn3 = MyKNeighborsClassifier(n_neighbors=3)
    my_knn3.fit(X_train_bramer_example, y_train_bramer_example)

    # create an unseen instance that we want to assign a label to.
    X_test3 = [[3, 10]]

    # generate a prediction.
    y_pred3 = my_knn3.predict(X_test3)

    expected_y_pred3 = ["-"]

    assert np.array_equiv(y_pred3, expected_y_pred3)



def test_dummy_classifier_fit():
    '''
        Purpose: test dummy classifier's fit() method.
    '''
    # set random seed for reproducibility.
    np.random.seed(0)

    # ==== test case 1 ====
    X_train1 = [[val] for val in range(0, 100)]
    y_train1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))

    # create a dummy classifier object and train it.
    my_classifier1 = MyDummyClassifier()
    my_classifier1.fit(X_train1, y_train1)

    assert my_classifier1.most_common_label == "yes"


    # ==== test case 2 ====
    X_train2 = [[val] for val in range(0, 100)]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))

    # create a dummy classifier object and train it.
    my_classifier2 = MyDummyClassifier()
    my_classifier2.fit(X_train2, y_train2)

    assert my_classifier2.most_common_label == "no"


    # ==== test case 3 ==== (labels evenly split)
    X_train3 = [[val] for val in range(0, 100)]
    y_train3 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.5, 0.5]))

    # create a dummy classifier object and train it.
    my_classifier3 = MyDummyClassifier()
    my_classifier3.fit(X_train3, y_train3)

    assert my_classifier3.most_common_label == "yes"


def test_dummy_classifier_predict():
    '''
        Purpose: test for dummy classifier output label predictions.
    '''
    # set random seed for reproducibility.
    np.random.seed(0)

    # ==== test case 1 ====
    X_train1 = [[val] for val in range(0, 100)]
    y_train1 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))

    # create a dummy classifier object and train it.
    my_classifier1 = MyDummyClassifier()
    my_classifier1.fit(X_train1, y_train1)

    X_test1 = [[1], [3], [6]]
    y_pred1 = my_classifier1.predict(X_test1)
    expected_pred1 = "yes"

    assert all(pred == expected_pred1 for pred in y_pred1)


    # ==== test case 2 ====
    X_train2 = [[val] for val in range(0, 100)]
    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    
    # create a dummy classifier object and train it.
    my_classifier2 = MyDummyClassifier()
    my_classifier2.fit(X_train2, y_train2)
    
    X_test2 = [[1], [3], [6]]
    y_pred2 = my_classifier2.predict(X_test2)
    expected_pred2 = "no"

    assert all(pred == expected_pred2 for pred in y_pred2)


    # ==== test case 3 ====
    X_train3 = [[val] for val in range(0, 100)]
    y_train3 = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.5, 0.5]))

    # create a dummy classifier object and train it.
    my_classifier3 = MyDummyClassifier()
    my_classifier3.fit(X_train3, y_train3)
    
    X_test3 = [[1], [3], [6]]
    y_pred3 = my_classifier3.predict(X_test3)
    expected_pred3 = "yes"

    assert all(pred == expected_pred3 for pred in y_pred3)


def test_naive_bayes_classifier_fit():
    '''
        Purpose: test for the naive bayes classification model training function.
    '''
    # case 1: use the 8-instance training set example (from class), asserting against our desk check of the priors and conditional probabilities.
    header1 = ["att1", "att2"]
    X_train1 = [
            [1, 5],
            [2, 6],
            [1, 5],
            [1, 5],
            [1, 6],
            [2, 6],
            [1, 5],
            [1, 6]
        ]
    y_train1 = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train

    # train the model.
    clf1 = MyNaiveBayesClassifier()
    clf1.fit(X_train1, y_train1)

    # check class priors. 
    assert np.isclose(clf1.priors["yes"], 5/8)
    assert np.isclose(clf1.priors["no"], 3/8)

    # check conditional probabilities.
    assert np.isclose(clf1.conditionals["yes"][0][1], 4/5)
    assert np.isclose(clf1.conditionals["yes"][0][2], 1/5)

    assert np.isclose(clf1.conditionals["yes"][1][5], 2/5)
    assert np.isclose(clf1.conditionals["yes"][1][6], 3/5)

    assert np.isclose(clf1.conditionals["no"][0][1], 2/3)
    assert np.isclose(clf1.conditionals["no"][0][2], 1/3)

    assert np.isclose(clf1.conditionals["no"][1][5], 2/3)
    assert np.isclose(clf1.conditionals["no"][1][6], 1/3)


    # case 2: use the 15 instance training set example (from LA7), asserting against your desk check of the priors and conditional probabilities.
    header2 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train2 = [
            [1, 3, "fair"],
            [1, 3, "excellent"], 
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]
        ]
    y_train2 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    
    clf2 = MyNaiveBayesClassifier()
    clf2.fit(X_train2, y_train2)

    # check class priors. 
    assert np.isclose(clf2.priors["yes"], 10/15)
    assert np.isclose(clf2.priors["no"], 5/15)

    # check conditional probabilities.
    assert np.isclose(clf2.conditionals["yes"][0][1], 2/10)
    assert np.isclose(clf2.conditionals["yes"][0][2], 8/10)
    assert np.isclose(clf2.conditionals["yes"][1][1], 3/10)
    assert np.isclose(clf2.conditionals["yes"][1][2], 4/10)
    assert np.isclose(clf2.conditionals["yes"][1][3], 3/10)
    assert np.isclose(clf2.conditionals["yes"][2]["fair"], 7/10)
    assert np.isclose(clf2.conditionals["yes"][2]["excellent"], 3/10)

    assert np.isclose(clf2.conditionals["no"][0][1], 3/5)
    assert np.isclose(clf2.conditionals["no"][0][2], 2/5)
    assert np.isclose(clf2.conditionals["no"][1][1], 1/5)
    assert np.isclose(clf2.conditionals["no"][1][2], 2/5)
    assert np.isclose(clf2.conditionals["no"][1][3], 2/5)
    assert np.isclose(clf2.conditionals["no"][2]["fair"], 2/5)
    assert np.isclose(clf2.conditionals["no"][2]["excellent"], 3/5)

    # case 3: Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and conditional probabilities in Figure 3.2.
    header3 = ["day", "season", "wind", "rain"]
    X_train3 = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "spring", "none", "none"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"],
    ]
    y_train3 = ["on time", "on time", "on time", "late", "on time", "very late", "on time", "on time", "very late", "on time", "cancelled", "on time", "late", "on time", "very late", "on time", "on time", "on time", "on time", "on time"]
    
    clf3 = MyNaiveBayesClassifier()
    clf3.fit(X_train3, y_train3)

    # check class priors. 
    assert np.isclose(clf3.priors["on time"], 0.70)
    assert np.isclose(clf3.priors["very late"], 0.15)
    assert np.isclose(clf3.priors["late"], 0.10)
    assert np.isclose(clf3.priors["cancelled"], 0.05)

    # check conditional probabilities.
    assert np.isclose(clf3.conditionals["on time"][0]["weekday"], 0.64, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][1]["winter"], 0.14, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][2]["high"], 0.29, atol=1e-2)
    assert np.isclose(clf3.conditionals["on time"][3]["heavy"], 0.07, atol=1e-2)

    assert np.isclose(clf3.conditionals["very late"][0]["weekday"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][1]["winter"], 0.67, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][2]["high"], 0.33, atol=1e-2)
    assert np.isclose(clf3.conditionals["very late"][3]["heavy"], 0.67, atol=1e-2)

    assert np.isclose(clf3.conditionals["late"][0]["weekday"], 0.5, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][1]["winter"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][2]["high"], 0.5, atol=1e-2)
    assert np.isclose(clf3.conditionals["late"][3]["heavy"], 0.5, atol=1e-2)

    # assert np.isclose(clf3.conditionals["cancelled"][0]["weekday"], 0)
    # assert np.isclose(clf3.conditionals["cancelled"][1]["winter"], 0)
    assert np.isclose(clf3.conditionals["cancelled"][2]["high"], 1, atol=1e-2)
    assert np.isclose(clf3.conditionals["cancelled"][3]["heavy"], 1, atol=1e-2)



def test_naive_bayes_classifier_predict():
    # case 1: use the 8-instance training set example (from class), asserting against our desk check of the priors and conditional probabilities.
    header1 = ["att1", "att2"]
    X_train1 = [
            [1, 5],
            [2, 6],
            [1, 5],
            [1, 5],
            [1, 6],
            [2, 6],
            [1, 5],
            [1, 6]
        ]
    y_train1 = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    X_test1 = [1, 5]

    # get predicted class label for our unseen instance using our naive bayes classifier.
    clf1 = MyNaiveBayesClassifier()
    clf1.fit(X_train1, y_train1)
    y_pred1 = clf1.predict(X_test1)

    # get true class label for our unseen instance using desk calculations.
    p_yes1 = 5/8
    p_1_given_yes = 4/5
    p_5_given_yes = 2/5
    p_yes_given_Xtest1 = p_yes1 * p_1_given_yes * p_5_given_yes
    
    p_no1 = 3/8
    p_1_given_no = 2/3
    p_5_given_no = 2/3
    p_no_given_Xtest1 = p_no1 * p_1_given_no * p_5_given_no

    if p_yes_given_Xtest1 > p_no_given_Xtest1:
        expected_class1 = "yes"
    else: # note: if the probabilite are the same, I just choose the label.
        expected_class1 = "no"
    
    assert expected_class1 == y_pred1


    # case 2: use the 15 instance training set example (from LA7), asserting against your desk check of the priors and conditional probabilities.
    header2 = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train2 = [
            [1, 3, "fair"],
            [1, 3, "excellent"], 
            [2, 3, "fair"],
            [2, 2, "fair"],
            [2, 1, "fair"],
            [2, 1, "excellent"],
            [2, 1, "excellent"],
            [1, 2, "fair"],
            [1, 1, "fair"],
            [2, 2, "fair"],
            [1, 2, "excellent"],
            [2, 2, "excellent"],
            [2, 3, "fair"],
            [2, 2, "excellent"],
            [2, 3, "fair"]
        ]
    y_train2 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    X_test2 = [
        [2, 2, "fair"], 
        [1, 1, "excellent"]
    ] 

    # get predicted class label for our unseen instances using our naive bayes classifier.
    clf2 = MyNaiveBayesClassifier()
    clf2.fit(X_train2, y_train2)

    y_pred2 = clf2.predict(X_test2)

    # get true class label for our unseen instances using (previously calculated) desk calculations.
    expected_label_Xtest2_1 = "yes" # from LA7
    expected_label_Xtest2_2 = "no" # from LA7

    assert y_pred2[0] == expected_label_Xtest2_1
    assert y_pred2[1] == expected_label_Xtest2_2

    

    # case 3: Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and conditional probabilities in Figure 3.2.
    header3 = ["day", "season", "wind", "rain"]
    X_train3 = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "spring", "none", "none"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"],
    ]
    y_train3 = ["on time", "on time", "on time", "late", "on time", "very late", "on time", "on time", "very late", "on time", "cancelled", "on time", "late", "on time", "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test3 = [
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "high", "heavy"],
        ["sunday", "summer", "normal", "slight"]
    ]

    # get predicted class label for our unseen instances using our naive bayes classifier.
    clf3 = MyNaiveBayesClassifier()
    clf3.fit(X_train3, y_train3)
    y_pred3 = clf3.predict(X_test3)

    # get true class label for our unseen instances using (previously calculated) desk calculations.
    expected_label_Xtest3_1 = "very late"
    expected_label_Xtest3_2 = "on time"
    expected_label_Xtest3_3 = "on time"

    assert expected_label_Xtest3_1 == y_pred3[0]
    assert expected_label_Xtest3_2 == y_pred3[1]
    assert expected_label_Xtest3_3 == y_pred3[2]