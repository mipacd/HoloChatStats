import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
from psycopg2.extras import execute_batch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import logging
from db.connection import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingDataset(Dataset):
    """Custom Dataset for streaming hours time series"""
    
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return max(0, len(self.data) - self.sequence_length)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting model"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class SimpleForecaster(nn.Module):
    """Simple MLP for channels with limited data"""
    
    def __init__(self, input_size: int, hidden_size: int = 32):
        super(SimpleForecaster, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.fc(x)


class StreamingHoursForecaster:
    """Main forecasting pipeline with adaptive model selection"""
    
    # Minimum data requirements
    MIN_MONTHS_LSTM = 12      # Use LSTM if we have at least 12 months
    MIN_MONTHS_SIMPLE = 4     # Use simple model if we have at least 4 months
    MIN_MONTHS_FALLBACK = 1   # Use statistical fallback for less data
    
    def __init__(self, db_conn, model_config: Optional[Dict] = None):
        self.db_conn = db_conn
        self.model_config = model_config or {
            'lstm_sequence_length': 6,   # Reduced from 12 to 6
            'simple_sequence_length': 3,  # For simple model
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 16,            # Reduced batch size
            'early_stopping_patience': 10
        }
        self.model_version = f"v1_{datetime.now().strftime('%Y%m%d')}"
        
    def connect_db(self):
        """Create database connection"""
        return self.db_conn
    
    def fetch_historical_data(self, channel_id: str = None) -> pd.DataFrame:
        """Fetch historical streaming data"""
        conn = self.connect_db()
        try:
            if channel_id:
                query = """
                    SELECT 
                        c.channel_id,
                        c.channel_name,
                        DATE_TRUNC('month', v.end_time)::DATE AS month, 
                        ROUND(CAST(SUM(EXTRACT(EPOCH FROM v.duration)) / 3600 AS NUMERIC), 2) AS total_streaming_hours
                    FROM videos v
                    JOIN channels c ON v.channel_id = c.channel_id
                    WHERE c.channel_id = %s
                    GROUP BY c.channel_id, c.channel_name, month
                    ORDER BY month;
                """
                df = pd.read_sql_query(query, conn, params=[channel_id])
            else:
                query = """
                    SELECT 
                        c.channel_id,
                        c.channel_name,
                        DATE_TRUNC('month', v.end_time)::DATE AS month, 
                        ROUND(CAST(SUM(EXTRACT(EPOCH FROM v.duration)) / 3600 AS NUMERIC), 2) AS total_streaming_hours
                    FROM videos v
                    JOIN channels c ON v.channel_id = c.channel_id
                    GROUP BY c.channel_id, c.channel_name, month
                    ORDER BY c.channel_id, month;
                """
                df = pd.read_sql_query(query, conn)
            
            df['month'] = pd.to_datetime(df['month'])
            return df
            
        finally:
            conn.close()
    
    def get_channel_data_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics about data availability per channel"""
        stats = df.groupby('channel_id').agg(
            months_of_data=('month', 'count'),
            first_month=('month', 'min'),
            last_month=('month', 'max'),
            avg_hours=('total_streaming_hours', 'mean'),
            std_hours=('total_streaming_hours', 'std')
        ).reset_index()
        return stats
    
    def prepare_channel_data(self, df: pd.DataFrame, channel_id: str) -> Tuple[np.ndarray, List[datetime], MinMaxScaler]:
        """Prepare data for a single channel"""
        channel_data = df[df['channel_id'] == channel_id].copy()
        channel_data = channel_data.sort_values('month')
        
        if len(channel_data) == 0:
            return None, None, None
        
        # Fill missing months with interpolation or 0
        date_range = pd.date_range(
            start=channel_data['month'].min(),
            end=channel_data['month'].max(),
            freq='MS'
        )
        
        channel_data = channel_data.set_index('month')
        channel_data = channel_data.reindex(date_range)
        
        # Interpolate missing values, then fill any remaining NaN with 0
        channel_data['total_streaming_hours'] = channel_data['total_streaming_hours'].interpolate(method='linear')
        channel_data['total_streaming_hours'] = channel_data['total_streaming_hours'].fillna(0)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Avoid exact 0 and 1
        values = channel_data['total_streaming_hours'].values.reshape(-1, 1).astype(float)
        
        # Handle edge case where all values are the same
        if values.std() == 0:
            values = values + np.random.normal(0, 0.01, values.shape)
        
        scaled_values = scaler.fit_transform(values)
        
        return scaled_values.flatten(), channel_data.index.tolist(), scaler
    
    def select_model_type(self, data_length: int) -> str:
        """Select appropriate model based on data availability"""
        if data_length >= self.MIN_MONTHS_LSTM:
            return 'lstm'
        elif data_length >= self.MIN_MONTHS_SIMPLE:
            return 'simple'
        elif data_length >= self.MIN_MONTHS_FALLBACK:
            return 'statistical'
        else:
            return 'none'
    
    def train_lstm_model(self, data: np.ndarray, channel_id: str) -> Tuple[LSTMForecaster, float]:
        """Train LSTM model for a single channel"""
        sequence_length = self.model_config['lstm_sequence_length']
        
        dataset = StreamingDataset(data, sequence_length)
        
        if len(dataset) < 2:
            return None, None
        
        # Split train/validation
        train_size = max(1, int(0.8 * len(dataset)))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=min(self.model_config['batch_size'], len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(self.model_config['batch_size'], max(1, len(val_dataset))), shuffle=False) if len(val_dataset) > 0 else None
        
        model = LSTMForecaster(
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.model_config['learning_rate'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.model_config['epochs']):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.unsqueeze(-1)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.unsqueeze(-1)
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
            else:
                avg_val_loss = train_loss / len(train_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.model_config['early_stopping_patience']:
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, best_val_loss
    
    def train_simple_model(self, data: np.ndarray, channel_id: str) -> Tuple[SimpleForecaster, float]:
        """Train simple MLP model for channels with limited data"""
        sequence_length = self.model_config['simple_sequence_length']
        
        dataset = StreamingDataset(data, sequence_length)
        
        if len(dataset) < 1:
            return None, None
        
        train_loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)
        
        model = SimpleForecaster(input_size=sequence_length, hidden_size=16)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(50):  # Fewer epochs for simple model
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.unsqueeze(-1)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, best_loss
    
    def statistical_forecast(self, data: np.ndarray, scaler: MinMaxScaler, n_months: int = 3) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Statistical forecast with P25 and P75"""
        if len(data) >= 3:
            weights = np.exp(np.linspace(-1, 0, len(data)))
            weights /= weights.sum()
            weighted_mean = np.average(data, weights=weights)
        else:
            weighted_mean = np.mean(data)
        
        std = np.std(data) if len(data) > 1 else 0.1
        
        if len(data) >= 2:
            trend = (data[-1] - data[0]) / len(data)
        else:
            trend = 0
        
        forecasts_scaled = []
        for i in range(n_months):
            forecast = weighted_mean + trend * (i + 1)
            forecast = np.clip(forecast, 0.05, 0.95)
            forecasts_scaled.append(forecast)
        
        forecasts_original = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()
        
        std_original = scaler.inverse_transform([[std]])[0][0] - scaler.inverse_transform([[0]])[0][0]
        
        # Calculate percentiles assuming normal distribution
        confidence_lower = [max(0, f - 1.645 * std_original * (1 + 0.2 * i)) for i, f in enumerate(forecasts_original)]
        confidence_p25 = [max(0, f - 0.674 * std_original * (1 + 0.2 * i)) for i, f in enumerate(forecasts_original)]
        confidence_p75 = [f + 0.674 * std_original * (1 + 0.2 * i) for i, f in enumerate(forecasts_original)]
        confidence_upper = [f + 1.645 * std_original * (1 + 0.2 * i) for i, f in enumerate(forecasts_original)]
        
        return (forecasts_original.tolist(), 
        confidence_lower, confidence_p25, confidence_p75, confidence_upper)
    
    def generate_forecasts(self, model: nn.Module, last_sequence: np.ndarray, 
                      scaler: MinMaxScaler, n_months: int = 3,
                      model_type: str = 'lstm') -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Generate forecasts with confidence intervals including P25 and P75"""
        model.eval()
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        n_simulations = 100  # Increased for better percentile estimates
        simulation_results = []
        
        for _ in range(n_months):
            month_simulations = []
            
            for _ in range(n_simulations):
                with torch.no_grad():
                    model.train()  # Enable dropout for uncertainty
                    input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                    prediction = model(input_tensor).item()
                    prediction = np.clip(prediction, 0, 1)
                    month_simulations.append(prediction)
            
            mean_prediction = np.mean(month_simulations)
            forecasts.append(mean_prediction)
            simulation_results.append(month_simulations)
            
            current_sequence = np.append(current_sequence[1:], mean_prediction)
        
        # Transform back to original scale
        forecasts_original = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        confidence_lower = []
        confidence_p25 = []
        confidence_p75 = []
        confidence_upper = []
        
        for simulations in simulation_results:
            simulations_original = scaler.inverse_transform(np.array(simulations).reshape(-1, 1)).flatten()
            confidence_lower.append(float(max(0, np.percentile(simulations_original, 5))))
            confidence_p25.append(float(max(0, np.percentile(simulations_original, 25))))
            confidence_p75.append(float(max(0, np.percentile(simulations_original, 75))))
            confidence_upper.append(float(np.percentile(simulations_original, 95)))
        
        return ([float(max(0, f)) for f in forecasts_original], 
                confidence_lower, confidence_p25, confidence_p75, confidence_upper)
    
    def save_forecasts(self, forecasts_data: List[Dict]):
        """Save forecasts to database with P25 and P75"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM streaming_forecasts 
                WHERE DATE(created_at) = CURRENT_DATE
            """)
            
            insert_query = """
                INSERT INTO streaming_forecasts 
                (channel_id, forecast_month, forecasted_hours, 
                confidence_lower, confidence_p25, confidence_p75, confidence_upper, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            execute_batch(cursor, insert_query, [
                (
                    f['channel_id'],
                    f['forecast_month'],
                    round(f['forecasted_hours'], 2),
                    round(f.get('confidence_lower', 0), 2),
                    round(f.get('confidence_p25', 0), 2),
                    round(f.get('confidence_p75', 0), 2),
                    round(f.get('confidence_upper', 0), 2),
                    f['model_version']
                )
                for f in forecasts_data
            ])
            
            conn.commit()
            logger.info(f"Saved {len(forecasts_data)} forecasts to database")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving forecasts: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def save_model_metrics(self, metrics_data: List[Dict]):
        """Save model performance metrics"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            insert_query = """
                INSERT INTO forecast_model_metrics 
                (channel_id, mae, rmse, mape, model_version)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            execute_batch(cursor, insert_query, [
                (
                    m['channel_id'],  # Keep as TEXT
                    m.get('mae'),
                    m.get('rmse'),
                    m.get('mape'),
                    m['model_version']
                )
                for m in metrics_data
            ])
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving metrics: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def run_forecasting_pipeline(self):
        """Main pipeline to train and forecast for all channels"""
        logger.info("Starting forecasting pipeline")
        
        # Fetch all historical data
        df = self.fetch_historical_data()
        
        if df.empty:
            logger.warning("No historical data found")
            return
        
        # Log data statistics
        stats = self.get_channel_data_stats(df)
        logger.info(f"Data statistics:\n{stats.describe()}")
        
        all_forecasts = []
        all_metrics = []
        
        processed = {'lstm': 0, 'simple': 0, 'statistical': 0, 'skipped': 0}
        
        for channel_id in df['channel_id'].unique():
            try:
                logger.info(f"Processing channel {channel_id}")
                
                # Prepare data
                scaled_data, dates, scaler = self.prepare_channel_data(df, channel_id)
                
                if scaled_data is None:
                    logger.warning(f"Skipping channel {channel_id} - no data")
                    processed['skipped'] += 1
                    continue
                
                data_length = len(scaled_data)
                model_type = self.select_model_type(data_length)
                
                logger.info(f"Channel {channel_id}: {data_length} months of data, using {model_type} model")
                
                forecasts = None
                conf_lower = None
                conf_p25 = None
                conf_p75 = None
                conf_upper = None
                val_loss = None
                
                if model_type == 'lstm':
                    model, val_loss = self.train_lstm_model(scaled_data, channel_id)
                    if model:
                        seq_len = self.model_config['lstm_sequence_length']
                        last_sequence = scaled_data[-seq_len:]
                        # Unpack all 5 values
                        forecasts, conf_lower, conf_p25, conf_p75, conf_upper = self.generate_forecasts(
                            model, last_sequence, scaler, n_months=3, model_type='lstm'
                        )
                        processed['lstm'] += 1
                    else:
                        model_type = 'simple'  # Fallback
                
                if model_type == 'simple':
                    model, val_loss = self.train_simple_model(scaled_data, channel_id)
                    if model:
                        seq_len = self.model_config['simple_sequence_length']
                        last_sequence = scaled_data[-seq_len:]
                        # Unpack all 5 values
                        forecasts, conf_lower, conf_p25, conf_p75, conf_upper = self.generate_forecasts(
                            model, last_sequence, scaler, n_months=3, model_type='simple'
                        )
                        processed['simple'] += 1
                    else:
                        model_type = 'statistical'  # Fallback
                
                if model_type == 'statistical':
                    # Unpack all 5 values
                    forecasts, conf_lower, conf_p25, conf_p75, conf_upper = self.statistical_forecast(
                        scaled_data, scaler, n_months=3
                    )
                    processed['statistical'] += 1
                
                if model_type == 'none' or forecasts is None:
                    logger.warning(f"Skipping channel {channel_id} - insufficient data for any model")
                    processed['skipped'] += 1
                    continue
                
                # Prepare forecast records with all confidence intervals
                last_date = dates[-1]
                for i in range(3):
                    forecast_month = last_date + relativedelta(months=i+1)
                    all_forecasts.append({
                        'channel_id': channel_id,
                        'forecast_month': forecast_month.date() if hasattr(forecast_month, 'date') else forecast_month,
                        'forecasted_hours': float(max(0, forecasts[i])),
                        'confidence_lower': float(max(0, conf_lower[i])),
                        'confidence_p25': float(max(0, conf_p25[i])),
                        'confidence_p75': float(max(0, conf_p75[i])),
                        'confidence_upper': float(max(0, conf_upper[i])),
                        'model_version': f"{self.model_version}_{model_type}"
                    })
                
                # Save metrics
                all_metrics.append({
                    'channel_id': channel_id,
                    'mae': float(np.sqrt(val_loss)) if val_loss else None,
                    'rmse': float(val_loss) if val_loss else None,
                    'mape': None,
                    'model_version': f"{self.model_version}_{model_type}"
                })
                
            except Exception as e:
                logger.error(f"Error processing channel {channel_id}: {e}", exc_info=True)
                processed['skipped'] += 1
                continue
        
        # Log summary
        logger.info(f"Processing summary: {processed}")
        
        # Save all forecasts and metrics
        if all_forecasts:
            self.save_forecasts(all_forecasts)
            logger.info(f"Generated {len(all_forecasts)} forecasts for {len(all_forecasts)//3} channels")
        
        if all_metrics:
            self.save_model_metrics(all_metrics)
        
        logger.info("Forecasting pipeline completed")
        
        return {
            'total_channels': len(df['channel_id'].unique()),
            'forecasts_generated': len(all_forecasts),
            'processing_summary': processed
        }


def main():
    forecaster = StreamingHoursForecaster(get_db_connection())
    result = forecaster.run_forecasting_pipeline()
    print(f"Pipeline result: {result}")


if __name__ == "__main__":
    main()