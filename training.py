import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import IsolationForest
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Training:
    
    def __init__(self, n_estimators, max_samples, contamination):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.model = None
    
    def train_model(self, X):
        isolation_forest_model = IsolationForest(n_estimators=self.n_estimators, 
                                                 max_samples=self.max_samples, 
                                                 contamination=self.contamination,
                                                 random_state=42, verbose=0)
        isolation_forest_model.fit(X)
        self.model = isolation_forest_model
        joblib.dump(self.model, 'isolation_forest_model.joblib')

def main():
    # Load your dataset here, replace 'your_dataset.csv' with your actual file path
    data = pd.read_csv("creditcard.csv")
    data1= data.sample(frac = 0.05,random_state=1)
    columns = data1.columns.tolist()
    columns = [c for c in columns if c not in ["Class"]]
    target = "Class"
    # Specify the features you want to use for training (X)
    # X = data[['feature1', 'feature2', 'feature3']]
    X = data1[columns]
    Y = data1[target]
    # Specify the hyperparameters for training
    n_estimators = 100
    max_samples = 'auto'
    contamination = 0.1

    # Create an instance of the Training class
    trainer = Training(n_estimators, max_samples, contamination)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the Isolation Forest model
    trainer.train_model(X_train)

    # Optionally, you can load the trained model and make predictions on the test set
    loaded_model = joblib.load('isolation_forest_model.joblib')
    y_pred = loaded_model.predict(X_test)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # Evaluate the accuracy
    accuracy = accuracy_score([1 if pred == -1 else 0 for pred in y_pred], [0] * len(y_pred))
    print(f"Accuracy is : {accuracy * 100:.2f}%")

    # Use Y_test for generating the classification report
    print("Classification Report :")
    print(classification_report(Y_test, y_pred))

    # Add any additional functionality you need in the main function

if __name__ == "__main__":
    main()
