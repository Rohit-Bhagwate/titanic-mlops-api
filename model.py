import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model():
    df = pd.read_csv("C:\\Users\\rohit\\Desktop\\MLOps\\MLOPs_Project\\titanic.csv")

    # Drop useless columns
    df.drop(["Name", "Ticket", "Cabin","PassengerId"], axis=1, inplace=True)

    # Convert categorical
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Fill missing values
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Split
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # Save
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model trained & saved")

# Run training when file executed
if __name__ == "__main__":
    train_model()