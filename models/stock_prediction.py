import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

# Thêm tham số stock_id để xử lý cả FPT và CMG
def train_stock_prediction_model(stock_id='FPT'):
    # 1. Đọc và tiền xử lý dữ liệu chính
    if stock_id == 'FPT':
        df = pd.read_csv("4.2.3 (TARGET) (live & his) FPT_detail_transactions_processed.csv")
    else:
        df = pd.read_csv("4.2.3 (TARGET) (live & his) CMG_detail_transactions_processed.csv")
    
    df = df[df['StockID'] == stock_id].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date')
    df['Closing Price'] = df['Closing Price'].str.replace(',', '').astype(float)
    df['Total Volume'] = df['Total Volume'].str.replace(',', '').astype(float)

    # 2. Tạo các đặc trưng cơ bản và nâng cao
    df['Return%'] = df['Closing Price'].pct_change() * 100
    df['MA5'] = df['Closing Price'].rolling(window=5).mean()
    df['MA10'] = df['Closing Price'].rolling(window=10).mean()
    df['Volume_ratio'] = df['Total Volume'] / df['Total Volume'].rolling(5).mean()
    df['Volatility'] = df['Closing Price'].pct_change().rolling(window=5).std() * 100
    df['Price_Momentum'] = df['Closing Price'].diff(5)
    df = df.fillna(0)

    # 3. Tích hợp dữ liệu sự kiện
    df_dividend = pd.read_csv("3.2 (live & his) news_dividend_issue (FPT_CMG)_processed.csv")
    df_meeting = pd.read_csv("3.3 (live & his) news_shareholder_meeting (FPT_CMG)_processed.csv")

    df_dividend_fpt = df_dividend[df_dividend['StockID'] == stock_id].copy()
    df_meeting_fpt = df_meeting[df_meeting['StockID'] == stock_id].copy()
    df_dividend_fpt.loc[:, 'Execution Date'] = pd.to_datetime(df_dividend_fpt['Execution Date'], format='%d/%m/%Y', errors='coerce')
    df_meeting_fpt.loc[:, 'Execution Date'] = pd.to_datetime(df_meeting_fpt['Execution Date'], format='%d/%m/%Y')

    df['Dividend_Event'] = df['Date'].isin(df_dividend_fpt['Execution Date']).astype(int)
    df['Meeting_Event'] = df['Date'].isin(df_meeting_fpt['Execution Date']).astype(int)

    # 4. Tích hợp dữ liệu tài chính
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

    df_financial_fpt = df_financial[(df_financial['Stocks'].str.contains(stock_id)) & (df_financial['Indicator'].isin(indicators))].copy()

    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    df_financial_melted = df_financial_fpt.melt(id_vars=['Indicator'], value_vars=quarters, var_name='Quarter', value_name='Value')
    quarter_dates = {
        'Q1_2023': '2023-01-01', 'Q2_2023': '2023-04-01', 'Q3_2023': '2023-07-01', 'Q4_2023': '2023-10-01',
        'Q1_2024': '2024-01-01', 'Q2_2024': '2024-04-01', 'Q3_2024': '2024-07-01', 'Q4_2024': '2024-10-01'
    }
    df_financial_melted['Date'] = df_financial_melted['Quarter'].map(quarter_dates)
    df_financial_melted['Date'] = pd.to_datetime(df_financial_melted['Date'])
    df_financial_pivot = df_financial_melted.pivot(index='Date', columns='Indicator', values='Value')

    df = df.merge(df_financial_pivot, left_on='Date', right_index=True, how='left')
    df[indicators] = df[indicators].ffill()
    df = df.dropna().reset_index(drop=True)

    # 5. Chuẩn bị dữ liệu cho mô hình
    features_xgb = ['Return%', 'MA5', 'MA10', 'Volume_ratio', 'Dividend_Event', 'Meeting_Event', 'Volatility', 'Price_Momentum'] + indicators
    X_xgb = df[features_xgb]
    y = df['Closing Price']

    # Chuẩn hóa dữ liệu
    y_log = np.log1p(df['Closing Price'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(y_log.values.reshape(-1, 1))
    lookback = 7
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled_data)):
        X_lstm.append(scaled_data[i - lookback:i, 0])
        y_lstm.append(scaled_data[i, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    X_xgb = X_xgb.iloc[lookback:].reset_index(drop=True)
    y = y.iloc[lookback:].reset_index(drop=True)

    # Loại bỏ NaN
    mask = ~(X_xgb.isna().any(axis=1) | y.isna())
    X_xgb = X_xgb[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    X_lstm = X_lstm[mask]
    y_lstm = y_lstm[mask]

    split = int(len(y) * 0.8)
    X_lstm_train, X_lstm_test = X_lstm[:split], X_lstm[split:]
    y_lstm_train, y_lstm_test = y_lstm[:split], y_lstm[split:]
    X_xgb_train, X_xgb_test = X_xgb[:split], X_xgb[split:]
    y_train, y_test = y[:split], y[split:]

    # 6. Huấn luyện mô hình LSTM
    model_lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(loss='mse', optimizer=Adam(0.001))
    model_lstm.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=16, verbose=0)
    pred_lstm_train = np.expm1(scaler.inverse_transform(model_lstm.predict(X_lstm_train))).flatten()
    pred_lstm_test = np.expm1(scaler.inverse_transform(model_lstm.predict(X_lstm_test))).flatten()

    pred_lstm_train = np.nan_to_num(pred_lstm_train, nan=np.nanmean(pred_lstm_train), neginf=0)
    pred_lstm_test = np.nan_to_num(pred_lstm_test, nan=np.nanmean(pred_lstm_test), neginf=0)

    # 7. Tối ưu và huấn luyện XGBoost
    param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    model_xgb = xgb.XGBRegressor()
    grid_search = GridSearchCV(model_xgb, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
    grid_search.fit(X_xgb_train, y_train)
    model_xgb = grid_search.best_estimator_
    pred_xgb_train = model_xgb.predict(X_xgb_train)
    pred_xgb_test = model_xgb.predict(X_xgb_test)

    # 8. Huấn luyện mô hình Meta
    X_meta_train = np.vstack((pred_lstm_train, pred_xgb_train)).T
    X_meta_test = np.vstack((pred_lstm_test, pred_xgb_test)).T
    meta_model = Sequential([
        Dense(64, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    meta_model.compile(optimizer=Adam(0.005), loss='mse')
    meta_model.fit(X_meta_train, y_train, epochs=150, batch_size=32, verbose=0)

    # 9. Đánh giá mô hình
    y_pred = meta_model.predict(X_meta_test).flatten()
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), neginf=0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    return meta_model, scaler, metrics, y_test, y_pred