import modelling as md
import pandas as pd
import feature_engineering as fe
import os
data = pd.read_csv(
<<<<<<< HEAD
    "technical_analysis_results.csv")  # Note, this the 10 years TECHNICAL INDICATORS DATA. Decide which one to use.
=======
    "data/data_1d_10y_stationary.csv")  # Note, this the 10 years TECHNICAL INDICATORS DATA. Decide which one to use.
>>>>>>> 63864d56eb5b97c2f2ff4d42a248a8a1e47a6f19

data = data.drop(["Date"], axis=1)
#data = fe.order_differencing(data, 1) #first order differencing
data = data.ffill(axis=0) #for stationary dataset only
#data = data.reset_index()
print(data)
xTrain, xTest, yTrain, yTest = fe.ordered_train_test_split(data, "Signal")

results = pd.DataFrame()
results["true_y"] = yTest

# results["prediction"] = md.fit_arima(data, "Close", len(yTest), True, ord=[2, 0, 2]).tolist()
# print("ARIMA Classifier")
# print(md.get_results(results["true_y"], results["prediction"]))

ada = md.fit_ada_boost(xTrain, yTrain, True)
results["prediction"] = ada.predict(xTest)
print("Adaboost Classifier")
print(results["prediction"])
print(md.get_results(results["true_y"], fe.generate_y(results, "prediction")))

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

rf = md.fit_random_forest(xTrain, yTrain)
results["prediction"] = rf.predict(xTest)
print("Random Forest Classifier")
print(md.get_results(results["true_y"], results["prediction"]))

lr = md.fit_logistic_regression(xTrain, yTrain)
results["prediction"] = lr.predict(xTest)
print("Logistic Regression")
print(md.get_results(results["true_y"], results["prediction"]))
print('Accuracy of the Logistic Regression on test set: {:.3f}'.format(lr.score(xTest, yTest)))

gb = md.fit_gradient_boosting(xTrain, yTrain, True)
results["prediction"] = gb.predict(xTest)
print("Gradient Boosting")
print(md.get_results(results["true_y"], results["prediction"]))
print('Accuracy of the GBM on test set: {:.3f}'.format(gb.score(xTest, yTest)))

xTrain_fnn, xTest_fnn, yTrain_fnn, yTest_fnn = md.reconstruct(xTrain, xTest, yTrain, yTest)
fnn = md.fit_FNN(xTrain_fnn, yTrain_fnn)
results["prediction"] = fnn.predict(xTest_fnn).flatten()
print("FNN")
print(md.get_results(yTest_fnn, results["prediction"]))
print('Accuracy of the FNN on test set: {:.3f}'.format(md.accuracy_score(yTest_fnn, results["prediction"])))

# data = pd.read_csv("data/data_minmax_1d_10y.csv")
# xTrain_seq, xTest_seq, yTrain_seq, yTest_seq, scaler_label = md.transform_and_rescale(data, history_size=4)
# model = md.fit_LSTM_reg(xTrain_seq, yTrain_seq)
# results = pd.DataFrame({"true_y": yTest_seq, "prediction": model.predict(xTest_seq).flatten()})
# print("LSTM Regressor")
# print('MSE of the LSTM on test set: {:.3f}'.format(md.get_mse(results['true_y'], results["prediction"], scaler_label)))

"""
Regression On Close Price
"""

data = pd.read_csv("data/data_reg_1d_10y.csv")

data = data.drop(["Date"], axis=1)
xTrain, xTest, yTrain, yTest = fe.ordered_train_test_split(data, "Y")
results = pd.DataFrame()
results["true_y"] = yTest

knn_reg_close = md.fit_KNN_reg(xTrain, yTrain, True)
results["prediction"] = knn_reg_close.predict(xTest)
results["trueSignal"] = fe.generate_y(results, "true_y")
results["signal"] = fe.generate_y(results, "prediction")
print("KNN Regressor")
print(md.get_results(results["trueSignal"], results["signal"]))

gradient_reg_close = md.fit_gradient_boosting_reg(xTrain, yTrain, True)
results["prediction"] = gradient_reg_close.predict(xTest)
results["trueSignal"] = fe.generate_y(results, "true_y")
results["signal"] = fe.generate_y(results, "prediction")
print("Gradient Boosting Regressor")
print(md.get_results(results["trueSignal"], results["signal"]))