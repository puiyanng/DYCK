import modelling as md
import pandas as pd
import feature_engineering as fe
import os
data = pd.read_csv(
    "data/data_normalized_1d_10y.csv")  # Note, this the 10 years TECHNICAL INDICATORS DATA. Decide which one to use.

data = data.drop(["Date"], axis=1)
data = data.drop(["Unnamed: 0"], axis=1)
signal = data["Signal"]
data = data.diff(periods=1) # we need to try without any differencing, with 1 period differencing and with 2 period differencing.
data["Signal"] = signal
print(data)
data = data.dropna()
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

data = pd.read_csv("data/data_minmax_1d_10y.csv")
xTrain_seq, xTest_seq, yTrain_seq, yTest_seq, scaler_label = md.transform_and_rescale(data, history_size=4)
model = md.fit_LSTM_reg(xTrain_seq, yTrain_seq)
results = pd.DataFrame({"true_y": yTest_seq, "prediction": model.predict(xTest_seq).flatten()})
print("LSTM Regressor")
print('MSE of the LSTM on test set: {:.3f}'.format(md.get_mse(results['true_y'], results["prediction"], scaler_label)))

