from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, X_test, y_train, y_test):
    """
    Train basic ML model
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"🎯 Model Accuracy: {accuracy}")
    return model
