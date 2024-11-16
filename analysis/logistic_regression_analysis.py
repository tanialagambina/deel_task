import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class LogisticRegressionModel:
    """
    Logistic Regression model class.
    """
    target_variable: str
    data: pd.DataFrame
    model: LogisticRegression
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def __init__(self, data: pd.DataFrame, target_variable: str):
        # Initialize with data and target variable
        self.data = data
        self.target_variable = target_variable

    def train_test_split_data(self):
        """
        Function to split the input data into the train and test X and y data,
        using the input target variable.
        """
        self.data.dropna(axis=1, inplace=True)
        self.X = self.data.drop(columns=[self.target_variable])  # Drop target and chargeback columns
        self.y = self.data[self.target_variable]

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2, # Rule of thumb test set size
            random_state=42
        )

    def train_model(self):
        """
        Function to train the LogisticRegression model with l1 regularization
        :return:
        """
        self.model = LogisticRegression(penalty='l1', solver='liblinear')
        self.model.fit(self.X_train, self.y_train)

    def extract_feature_importance(self) -> pd.DataFrame:
        """
        Function to extract the calculated feature importances from the trained model
        """
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': np.abs(coefficients)
        })

        return feature_importance.sort_values(by='importance', ascending=False)

    def evaluate_model(self):
        """
        Function to determine the performance of the model on the separate test set.
        Function produces a confusion matrix and ROC curve to visualise performance.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        gradient_palette = sns.color_palette("Purples", as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=gradient_palette, xticklabels=['Declined', 'Accepted'],
                    yticklabels=['Declined', 'Accepted'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()