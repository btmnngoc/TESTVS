import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_paths(stock_id):
    return {
        "meta_model": os.path.join(MODEL_DIR, f"{stock_id}_meta_model.h5"),
        "lstm_model": os.path.join(MODEL_DIR, f"{stock_id}_lstm_model.h5"),
        "xgb_model": os.path.join(MODEL_DIR, f"{stock_id}_xgb_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, f"{stock_id}_scaler.pkl"),
        "lookback": os.path.join(MODEL_DIR, f"{stock_id}_lookback.pkl"),
        "features": os.path.join(MODEL_DIR, f"{stock_id}_features.pkl")
    }

def save_trained_models(stock_id, models, scaler, lookback, features):
    paths = get_model_paths(stock_id)
    save_model(models["meta"], paths["meta_model"])
    save_model(models["lstm"], paths["lstm_model"])
    joblib.dump(models["xgb"], paths["xgb_model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(lookback, paths["lookback"])
    joblib.dump(features, paths["features"])

def load_trained_models(stock_id):
    paths = get_model_paths(stock_id)
    try:
        models = {
            "meta": load_model(paths["meta_model"]),
            "lstm": load_model(paths["lstm_model"]),
            "xgb": joblib.load(paths["xgb_model"])
        }
        scaler = joblib.load(paths["scaler"])
        lookback = joblib.load(paths["lookback"])
        features = joblib.load(paths["features"])
        return models, scaler, lookback, features
    except FileNotFoundError:
        return None, None, None, None

def prepare_data(stock_id):
    # Load transaction data
    if stock_id == 'FPT':
        df = pd.read_csv("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    else:
        df = pd.read_csv("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    
    df = df[df['StockID'] == stock_id].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
    df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)

    # Feature engineering
    df['Return%'] = df['Closing Price'].pct_change() * 100
    df['MA5'] = df['Closing Price'].rolling(window=5).mean()
    df['MA10'] = df['Closing Price'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Total Volume'] / df['Total Volume'].rolling(5).mean()
    df['Volatility'] = df['Closing Price'].pct_change().rolling(window=5).std() * 100
    df['Price_Momentum'] = df['Closing Price'].diff(5)
    df = df.fillna(0)

    # Load and merge event data
    df_dividend = pd.read_csv("3.2 (live & his) news_dividend_issue (FPT_CMG)_processed.csv")
    df_meeting = pd.read_csv("3.3 (live & his) news_shareholder_meeting (FPT_CMG)_processed.csv")

    df_dividend = df_dividend[df_dividend['StockID'] == stock_id].copy()
    df_meeting = df_meeting[df_meeting['StockID'] == stock_id].copy()
    df_dividend.loc[:, 'Execution Date'] = pd.to_datetime(df_dividend['Execution Date'], format='%d/%m/%Y', errors='coerce')
    df_meeting.loc[:, 'Execution Date'] = pd.to_datetime(df_meeting['Execution Date'], format='%d/%m/%Y')

    df['Dividend_Event'] = df['Date'].isin(df_dividend['Execution Date']).astype(int)
    df['Meeting_Event'] = df['Date'].isin(df_meeting['Execution Date']).astype(int)

    # Load and merge financial data
    df_financial = pd.read_csv("6.5 (his) financialreport_metrics_Nhóm ngành_Công nghệ thông tin (of FPT_CMG)_processed.csv")
    
    def clean_financial_data(df):
        df['Indicator'] = df['Indicator'].str.replace('\n', '', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
        for col in df.columns[3:]:
            df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
        return df
    
    df_financial = clean_financial_data(df_financial)

    indicators = [
        'Tỷ suất lợi nhuận trên Vốn chủ sở hữu bình quân (ROEA)%',
        'Tỷ lệ lãi EBIT%',
        'Chỉ số giá thị trường trên giá trị sổ sách (P/B)Lần',
        'Chỉ số giá thị trường trên thu nhập (P/E)Lần',
        'P/SLần',
        'Tỷ suất sinh lợi trên vốn dài hạn bình quân (ROCE)%',
        'Thu nhập trên mỗi cổ phần (EPS)VNĐ'
    ]

    df_financial = df_financial[(df_financial['Stocks'].str.contains(stock_id)) & (df_financial['Indicator'].isin(indicators))].copy()

    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    df_financial = df_financial.melt(id_vars=['Indicator'], value_vars=quarters, var_name='Quarter', value_name='Value')
    
    quarter_dates = {
        'Q1_2023': '2023-01-01', 'Q2_2023': '2023-04-01', 'Q3_2023': '2023-07-01', 'Q4_2023': '2023-10-01',
        'Q1_2024': '2024-01-01', 'Q2_2024': '2024-04-01', 'Q3_2024': '2024-07-01', 'Q4_2024': '2024-10-01'
    }
    df_financial['Date'] = df_financial['Quarter'].map(quarter_dates)
    df_financial['Date'] = pd.to_datetime(df_financial['Date'])
    df_financial = df_financial.pivot(index='Date', columns='Indicator', values='Value')

    df = df.merge(df_financial, left_on='Date', right_index=True, how='left')
    df[indicators] = df[indicators].ffill()
    df = df.dropna().reset_index(drop=True)

    return df, indicators

def train_lstm_model(X_train, y_train, lookback):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(0.001))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    return model

def train_xgboost_model(X_train, y_train):
    param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_meta_model(X_train, y_train):
    model = Sequential([
        Dense(64, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.005), loss='mse')
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
    return model

def train_stock_prediction_model(stock_id='FPT', force_retrain=False):
    # Try to load existing models first
    if not force_retrain:
        models, scaler, lookback, features = load_trained_models(stock_id)
        if models is not None:
            # Prepare evaluation data
            df, indicators = prepare_data(stock_id)
            features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 
                          'Meeting_Event', 'Volatility', 'Price_Momentum'] + indicators
            X_xgb = df[features_xgb]
            y = df['Closing Price']
            
            # Scale data for LSTM
            y_log = np.log1p(y)
            scaled_data = scaler.transform(y_log.values.reshape(-1, 1))
            
            # Prepare LSTM inputs
            X_lstm, y_lstm = [], []
            for i in range(lookback, len(scaled_data)):
                X_lstm.append(scaled_data[i-lookback:i, 0])
                y_lstm.append(scaled_data[i, 0])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
            
            # Align data
            X_xgb = X_xgb.iloc[lookback:].reset_index(drop=True)
            y = y.iloc[lookback:].reset_index(drop=True)
            
            # Remove NaN
            mask = ~(X_xgb.isna().any(axis=1) | y.isna())
            X_xgb = X_xgb[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
            X_lstm = X_lstm[mask]
            y_lstm = y_lstm[mask]
            
            # Split data
            split = int(len(y) * 0.8)
            X_lstm_test = X_lstm[split:]
            y_test = y[split:]
            
            # Make predictions
            pred_lstm_test = np.expm1(scaler.inverse_transform(models["lstm"].predict(X_lstm_test))).flatten()
            pred_xgb_test = models["xgb"].predict(X_xgb.iloc[split:])
            
            # Combine predictions
            X_meta_test = np.vstack((pred_lstm_test, pred_xgb_test)).T
            y_pred = models["meta"].predict(X_meta_test).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'last_training_date': df['Date'].max().strftime('%Y-%m-%d')
            }
            
            return models, scaler, lookback, metrics, y_test, y_pred, df
    
    # If models don't exist or force_retrain=True, train new models
    df, indicators = prepare_data(stock_id)
    features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 
                   'Meeting_Event', 'Volatility', 'Price_Momentum'] + indicators
    X_xgb = df[features_xgb]
    y = df['Closing Price']
    
    # Scale data for LSTM
    lookback = 7
    y_log = np.log1p(y)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(y_log.values.reshape(-1, 1))
    
    # Prepare LSTM inputs
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled_data)):
        X_lstm.append(scaled_data[i-lookback:i, 0])
        y_lstm.append(scaled_data[i, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    # Align data
    X_xgb = X_xgb.iloc[lookback:].reset_index(drop=True)
    y = y.iloc[lookback:].reset_index(drop=True)
    
    # Remove NaN
    mask = ~(X_xgb.isna().any(axis=1) | y.isna())
    X_xgb = X_xgb[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    X_lstm = X_lstm[mask]
    y_lstm = y_lstm[mask]
    
    # Split data
    split = int(len(y) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
    y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]
    X_xgb_train, X_xgb_test = X_xgb[:split], X_xgb[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train models
    lstm_model = train_lstm_model(X_lstm_train, y_lstm_train, lookback)
    xgb_model = train_xgboost_model(X_xgb_train, y_train)
    
    # Make base predictions
    pred_lstm_train = np.expm1(scaler.inverse_transform(lstm_model.predict(X_lstm_train))).flatten()
    pred_lstm_test = np.expm1(scaler.inverse_transform(lstm_model.predict(X_lstm_test))).flatten()
    pred_xgb_train = xgb_model.predict(X_xgb_train)
    pred_xgb_test = xgb_model.predict(X_xgb_test)
    
    # Handle NaN values
    pred_lstm_train = np.nan_to_num(pred_lstm_train, nan=np.nanmean(pred_lstm_train), neginf=0)
    pred_lstm_test = np.nan_to_num(pred_lstm_test, nan=np.nanmean(pred_lstm_test), neginf=0)
    
    # Train meta model
    X_meta_train = np.vstack((pred_lstm_train, pred_xgb_train)).T
    X_meta_test = np.vstack((pred_lstm_test, pred_xgb_test)).T
    meta_model = train_meta_model(X_meta_train, y_train)
    
    # Make final predictions
    y_pred = meta_model.predict(X_meta_test).flatten()
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), neginf=0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'last_training_date': df['Date'].max().strftime('%Y-%m-%d')
    }
    
    # Save models
    models = {
        "meta": meta_model,
        "lstm": lstm_model,
        "xgb": xgb_model
    }
    save_trained_models(stock_id, models, scaler, lookback, features_xgb)
    
    return models, scaler, lookback, metrics, y_test, y_pred, df