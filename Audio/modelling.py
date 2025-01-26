import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def train_model():
    data = pd.read_csv("Output/features.csv")
    X = data.iloc[:, :-1].values
    Y = data["labels"].values

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 2, 3, 4, 5],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 0.1, 0.01, 0],
    }

    model = XGBClassifier(random_state=1, verbosity=0)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=1,
    )

    search.fit(x_train, y_train)

    print("Best Parameters:", search.best_params_)
    print("\nClassification Report:")
    y_pred = search.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("Output/confusion_matrix.png")
    plt.close()

    joblib.dump(search.best_estimator_, "Models/model.pkl")
    joblib.dump(scaler, "Models/scaler.pkl")
    joblib.dump(label_encoder, "Models/label_encoder.pkl")


if __name__ == "__main__":
    train_model()
