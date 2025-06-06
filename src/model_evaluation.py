from sklearn.metrics import classification_report, confusion_matrix
import pickle

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
