import pandas as pd
import modelling as md
import feature_engineering as fe
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "data/data_normalized_1d_10y.csv")  # Note, this the 10 years data file. Decide which one to use.

data = data.drop(["Date"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(data.drop(["Signal"], axis=1),
                                                data["Signal"], test_size=0.2,
                                                random_state=42)

prediction = pd.DataFrame()
results = pd.DataFrame()
results["true_y"] = yTest

test_lstm = True

if not test_lstm:
    ada = md.fit_ada_boost(xTrain, yTrain)
    results["prediction"] = ada.predict(xTest)
    print("Adaboost Classifier")
    print(md.get_results(results["true_y"], results["prediction"]))
    print('Accuracy of the Adaboost on test set: {:.3f}'.format(ada.score(xTest, yTest)))

    rf = md.fit_random_forest(xTrain, yTrain)
    results["prediction"] = rf.predict(xTest)
    print("Random Forest Classifier")
    print(md.get_results(results["true_y"], results["prediction"]))
    print('Accuracy of the Random Forest on test set: {:.3f}'.format(rf.score(xTest, yTest)))

    svm = md.fit_SVM(xTrain, yTrain)
    results["prediction"] = svm.predict(xTest)
    print("SVM Classifier")
    print(md.get_results(results["true_y"], results["prediction"]))
    print('Accuracy of the SVM on test set: {:.3f}'.format(svm.score(xTest, yTest)))

    knn = md.fit_KNN(xTrain, yTrain, True)
    results["prediction"] = knn.predict(xTest)
    print("KNN")
    print(md.get_results(results["true_y"], results["prediction"]))
    print('Accuracy of the KNN on test set: {:.3f}'.format(knn.score(xTest, yTest)))

    gb = md.fit_gradient_boosting(xTrain, yTrain, True)
    results["prediction"] = gb.predict(xTest)
    print("Gradient Boosting")
    print(md.get_results(results["true_y"], results["prediction"]))
    print('Accuracy of the GBM on test set: {:.3f}'.format(gb.score(xTest, yTest)))

lstm = md.fit_LSTM(xTrain, yTrain, True)
# results["prediction"] = lstm.predict(xTest)
# print("Gradient Boosting")
# print(lstm.get_results(results["true_y"], results["prediction"]))
# print('Accuracy of the GBM on test set: {:.3f}'.format(lstm.score(xTest, yTest)))
