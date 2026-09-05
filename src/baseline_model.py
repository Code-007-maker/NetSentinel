from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class LRBaseline:
    """
    Logistic Regression baseline that operates on the aggregated features 
    of a time window, ensuring a fair comparison to the World Model.
    """
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def extract_features(self, df_window: pd.DataFrame) -> np.ndarray:
        """
        Extracts equivalent current-window features for the LR model.
        We aggregate flow statistics in the window.
        """
        if df_window.empty:
            return np.zeros(5)
            
        features = [
            df_window['duration'].mean(),
            df_window['packets_forward'].sum(),
            df_window['packets_backward'].sum(),
            df_window['bytes_forward'].sum(),
            df_window['bytes_backward'].sum()
        ]
        return np.array(features)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the logistic regression model."""
        self.model.fit(X_train, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
