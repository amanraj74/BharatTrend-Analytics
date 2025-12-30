"""
Demand Forecasting Module
Uses moving average and linear regression for time series forecasting
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_config, setup_logging, load_data, save_checkpoint

logger = setup_logging()

class DemandForecaster:
    """
    Forecast product demand using sentiment trends
    """
    
    def __init__(self, config: Dict):
        """
        Initialize forecaster
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.forecast_config = config['forecasting']
        self.model = LinearRegression()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for forecasting
        
        Args:
            df: Input dataframe with date and sentiment
            
        Returns:
            DataFrame with features
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Time-based features
        df['day_num'] = (df['date'] - df['date'].min()).dt.days
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Moving averages
        df['ma_7'] = df['avg_sentiment'].rolling(window=7, min_periods=1).mean()
        df['ma_14'] = df['avg_sentiment'].rolling(window=14, min_periods=1).mean()
        
        # Lag features
        df['lag_1'] = df['avg_sentiment'].shift(1)
        df['lag_7'] = df['avg_sentiment'].shift(7)
        
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> LinearRegression:
        """
        Train forecasting model
        
        Args:
            df: DataFrame with features
            
        Returns:
            Trained model
        """
        logger.info("Training forecasting model...")
        
        feature_cols = ['day_num', 'day_of_week', 'ma_7', 'ma_14', 'lag_1', 'lag_7']
        X = df[feature_cols]
        y = df['avg_sentiment']
        
        self.model.fit(X, y)
        
        logger.info("✓ Model trained successfully")
        return self.model
    
    def make_forecast(self, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        Make future predictions
        
        Args:
            df: Historical data
            periods: Number of days to forecast
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Forecasting {periods} days ahead...")
        
        last_date = df['date'].max()
        last_values = df.iloc[-14:].copy()  # Use last 14 days
        
        predictions = []
        
        for i in range(periods):
            future_date = last_date + pd.Timedelta(days=i+1)
            day_num = (future_date - df['date'].min()).days
            
            # Create features for prediction
            features = {
                'day_num': day_num,
                'day_of_week': future_date.dayofweek,
                'ma_7': last_values['avg_sentiment'].tail(7).mean(),
                'ma_14': last_values['avg_sentiment'].tail(14).mean(),
                'lag_1': last_values['avg_sentiment'].iloc[-1],
                'lag_7': last_values['avg_sentiment'].iloc[-7] if len(last_values) >= 7 else last_values['avg_sentiment'].mean()
            }
            
            X_pred = pd.DataFrame([features])
            y_pred = self.model.predict(X_pred)[0]
            
            predictions.append({
                'date': future_date,
                'predicted_sentiment': y_pred,
                'forecast_type': 'prediction'
            })
            
            # Update last_values for next iteration
            new_row = pd.DataFrame({
                'date': [future_date],
                'avg_sentiment': [y_pred],
                'day_num': [day_num],
                'day_of_week': [future_date.dayofweek],
                'ma_7': [features['ma_7']],
                'ma_14': [features['ma_14']],
                'lag_1': [features['lag_1']],
                'lag_7': [features['lag_7']]
            })
            last_values = pd.concat([last_values, new_row], ignore_index=True)
        
        forecast_df = pd.DataFrame(predictions)
        logger.info("✓ Forecast completed")
        return forecast_df
    
    def evaluate_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Train model
        self.train_model(train_df)
        
        # Predict on test set
        feature_cols = ['day_num', 'day_of_week', 'ma_7', 'ma_14', 'lag_1', 'lag_7']
        X_test = test_df[feature_cols]
        y_test = test_df['avg_sentiment']
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        logger.info(f"Model Evaluation:")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_ratio = self.forecast_config['train_test_split']
        split_idx = int(len(df) * split_ratio)
        
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train)} points, Test: {len(test)} points")
        return train, test

def main():
    """
    Main execution - train forecasting model
    """
    logger.info("="*60)
    logger.info("BharatTrend Demand Forecasting Pipeline")
    logger.info("="*60)
    
    config = load_config()
    forecaster = DemandForecaster(config)
    
    # Load sentiment trends data
    trends_path = f"{config['paths']['processed_data']}/sentiment_trends.csv"
    
    if not Path(trends_path).exists():
        logger.warning(f"Sentiment trends not found: {trends_path}")
        logger.info("Run sentiment analysis first")
        return
    
    logger.info(f"\nLoading data from: {trends_path}")
    df = load_data(trends_path)
    
    if df is not None and not df.empty:
        # Prepare features
        logger.info("Preparing features...")
        df_features = forecaster.prepare_features(df)
        
        # Split data
        train_df, test_df = forecaster.train_test_split(df_features)
        
        # Train and evaluate
        logger.info("\nTraining and evaluating model...")
        metrics = forecaster.evaluate_model(train_df, test_df)
        
        # Train on full data
        logger.info("\nTraining on full dataset...")
        forecaster.train_model(df_features)
        
        # Make forecast
        forecast = forecaster.make_forecast(df_features, periods=14)
        
        # Save forecast
        output_path = f"{config['paths']['processed_data']}/demand_forecast.csv"
        save_checkpoint(forecast, output_path)
        
        # Also save combined historical + forecast
        historical = df[['date', 'avg_sentiment']].copy()
        historical['forecast_type'] = 'historical'
        historical.rename(columns={'avg_sentiment': 'predicted_sentiment'}, inplace=True)
        
        combined = pd.concat([historical, forecast], ignore_index=True)
        combined_path = f"{config['paths']['processed_data']}/full_forecast.csv"
        save_checkpoint(combined, combined_path)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Forecasting pipeline completed")
        logger.info(f"✓ Forecast saved: {output_path}")
        logger.info("="*60)
    else:
        logger.error("Failed to load data")

if __name__ == "__main__":
    main()
