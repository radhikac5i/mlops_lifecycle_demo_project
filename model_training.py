import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from datetime import datetime

class DataPreprocessing:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        self.model: Optional[XGBClassifier] = None
        self.label_encoders: dict = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.load_data()

    def load_data(self) -> None:
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")

    def handle_missing_values(self, num_strategy: str = "mean", cat_strategy: str = "mode") -> None:
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns

        for feature in numeric_features:
            if self.data[feature].isnull().sum() > 0:
                fill_value = self.data[feature].mean() if num_strategy == "mean" else self.data[feature].median()
                self.data[feature].fillna(fill_value, inplace=True)

        for feature in categorical_features:
            if self.data[feature].isnull().sum() > 0:
                self.data[feature].fillna(self.data[feature].mode()[0], inplace=True)

        print("Missing values handled successfully.")

    def detect_outliers(self, features: List[str], z_thresh: float = 3.0) -> pd.DataFrame:
        outliers = pd.DataFrame()
        for feature in features:
            if feature in self.data.columns:
                z_scores = np.abs(stats.zscore(self.data[feature].dropna()))
                feature_outliers = self.data.loc[self.data[feature].dropna().index[z_scores > z_thresh]]
                outliers = pd.concat([outliers, feature_outliers])
        print(f"Outliers detected in features: {features}")
        return outliers.drop_duplicates()

    def remove_outliers(self, features: List[str], z_thresh: float = 3.0) -> None:
        print("Before outlier removal")
        print(self.data.describe())
        condition = np.ones(len(self.data), dtype=bool)
        for feature in features:
            if feature in self.data.columns:
                z_scores = np.abs(stats.zscore(self.data[feature].dropna()))
                clean_idx = self.data[feature].dropna().index[z_scores <= z_thresh]
                condition = condition & self.data.index.isin(clean_idx)
        self.data = self.data.loc[condition]
        print("After outlier removal")
        print(self.data.describe())

    def drop_columns(self, columns: List[str]) -> None:
        self.data.drop(columns=[col for col in columns if col in self.data.columns], inplace=True)
        print(f"Columns {columns} dropped successfully.")

    def encode_categorical_features(self) -> None:
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns
        for feature in categorical_features:
            le = LabelEncoder()
            self.data[feature] = le.fit_transform(self.data[feature])
            self.label_encoders[feature] = le
        print(f"Categorical features {list(categorical_features)} encoded successfully.")

    def split_data(self, target_column: str, test_size: float = 0.2, random_state: int = 42) -> None:
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("y_test distribution:", self.y_test.value_counts())
        print(f"Data split into train and test sets with test size = {test_size}.")

    def train_xgboost(self) -> None:

        print("Before SMOTE class distribution:")
        print(self.y_train.value_counts())
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        print("After SMOTE class distribution:")
        print(pd.Series(y_resampled).value_counts())

        train_combined = pd.concat([X_resampled, y_resampled.rename("Exited")], axis=1)
        train_csv_path = "train_data.csv"
        train_combined.to_csv(train_csv_path, index=False)
        

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Create timestamped run name
        run_name = f"churn_xgb_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # Start a new MLflow run
        # mlflow.start_run(run_name=run_name)
        # mlflow.start_run(run_name="XGBoost_Churn_Model1")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model", "XGBoost")
            mlflow.set_tag("preprocessing", "SMOTE")

            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            self.model.fit(X_resampled, y_resampled)

            mlflow.log_param("use_label_encoder", False)
            mlflow.log_param("eval_metric", "logloss")
            mlflow.log_param("random_state", 42)

            print("XGBoost model trained successfully using SMOTE-balanced data.")
            mlflow.log_artifact(train_csv_path)
            self.evaluate_model()
        mlflow.end_run()

    def evaluate_model(self) -> None:
        test_combined = pd.concat([self.X_test, self.y_test.rename("Exited")], axis=1)
        test_csv_path = "test_data.csv"
        test_combined.to_csv(test_csv_path, index=False)
        mlflow.log_artifact(test_csv_path)
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        print("✅ Evaluation Metrics")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall   : {recall:.2f}")
        print(f"F1-Score : {f1:.2f}")
        print("\\n🔍 Classification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(self.model, "xgboost_churn_model")

        mlflow.register_model(
            model_uri="runs:/" + mlflow.active_run().info.run_id + "/xgboost_churn_model",
            name="XGBoostCustomerChurnModel"
        )

        

    def plot_roc_auc(self, save_path: str = 'roc_auc_curve.png') -> None:
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"ROC curve plotted and saved to {save_path}.")

    def save_model(self, save_path: str = 'xgboost_model.pkl') -> None:
        joblib.dump(self.model, save_path)
        print(f"Model saved successfully at {save_path}.")

def main():
    dp = DataPreprocessing(filepath='data_store/feast_training_data.csv')
    dp.drop_columns(columns=['CustomerID', 'event_timestamp'])
    dp.handle_missing_values(num_strategy='mean', cat_strategy='mode')
    dp.encode_categorical_features()
    dp.detect_outliers(features=['Age', 'Balance'])
    dp.remove_outliers(features=['Age', 'Balance'])
    dp.split_data(target_column='Exited', test_size=0.2)
    dp.train_xgboost()
    # dp.evaluate_model()
    dp.plot_roc_auc(save_path='model_output/roc_auc_curve.png')
    dp.save_model(save_path='model_output/xgboost_churn_model.pkl')
    
if __name__ == "__main__":
    main()