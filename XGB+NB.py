import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, BatchNormalization, Reshape, Activation, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 设置 Matplotlib 字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义公建机组列表
PUBLIC_BUILDINGS = [
    "XXX", "XXXX"
]

def build_nb_beats_model(input_shape):
    """构建带有自注意力机制的N-BEATS模型"""
    inputs = Input(shape=input_shape)

    def block(x, dilation_rate):
        x = Conv1D(filters=128, kernel_size=3, padding='causal', dilation_rate=dilation_rate,
                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        return x

    x = Reshape((input_shape[0], 1))(inputs)
    for i in range(4):
        x = block(x, dilation_rate=2 ** i)

    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def prepare_features_for_public_building(df, current_start, current_end, weather_data, resample_period):
    """为公建准备特征"""
    segment_hours = int(resample_period[0])
    periods_per_day = 24 // segment_hours

    prev_same_period_start = current_start - timedelta(days=1)
    prev_same_period_end = prev_same_period_start + timedelta(hours=segment_hours)
    same_period_prev_day = df.loc[prev_same_period_start:prev_same_period_end]['SST'].mean()

    prev_period_start = current_start - timedelta(hours=2 * segment_hours)
    prev_period_end = prev_period_start + timedelta(hours=segment_hours)
    prev_period_load = df.loc[prev_period_start:prev_period_end]['SST'].mean()
    prev_period_srt = df.loc[prev_period_start:prev_period_end]['SRT'].mean()

    future_weather = weather_data.loc[current_start:current_end]
    forecast_temp = future_weather['item_temperature'].mean()
    forecast_dswrf = future_weather['item_dswrf'].mean()

    return np.array([
        same_period_prev_day,
        prev_period_load,
        prev_period_srt,
        forecast_temp,
        forecast_dswrf
    ])

def process_data(file_path, resample_period):
    print(f"正在处理数据: {file_path}")
    df = pd.read_excel(file_path)

    df['ts_interval_x'] = pd.to_datetime(df['ts_interval_x'], errors='coerce')
    df = df.dropna(subset=['ts_interval_x'])
    df.set_index('ts_interval_x', inplace=True)

    required_columns = ['SST', 'SRT', 'item_temperature', 'item_dswrf']
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].ffill().bfill()

    df = df.resample(resample_period).mean()
    return df

def load_weather_data(file_path):
    print(f"正在加载天气数据: {file_path}")
    df = pd.read_csv(file_path)
    df['ts_interval_x'] = pd.to_datetime(df['ts_interval_x'])
    df.set_index('ts_interval_x', inplace=True)

    for col in ['item_temperature', 'item_dswrf']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def create_cyclic_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df

def limit_change_rate(predicted, previous, change_rate_limit):
    change_rate = abs(predicted - previous) / previous * 100
    if change_rate > change_rate_limit:
        if predicted > previous:
            predicted = previous * (1 + change_rate_limit / 100)
        else:
            predicted = previous * (1 - change_rate_limit / 100)
    return predicted

def limit_prediction(predicted, params):
    return max(min(predicted, params['flowUpperLimit']), params['flowLowerLimit'])

def prepare_features_for_other(df, current_start, last_SST, last_avg, weather_data, current_end):
    """为非公建准备特征"""
    future_weather = weather_data.loc[current_start:current_end]
    current_dswrf = future_weather['item_dswrf'].mean()
    current_temp = future_weather['item_temperature'].mean()

    return np.array([
        last_SST[-4], last_SST[-3], last_SST[-2], last_SST[-1],
        last_avg, current_dswrf, current_temp
    ])

def evaluate_model(y_true, y_pred, model_name="综合模型"):
    """计算并输出模型评价指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} 评价指标:")
    print(f"  - 均方误差 (MSE): {mse:.4f}")
    print(f"  - 均方根误差 (RMSE): {rmse:.4f}")
    print(f"  - 平均绝对误差 (MAE): {mae:.4f}")
    print(f"  - 决定系数 (R²): {r2:.4f}")

def plot_performance(y_true, y_pred, building_name, period_index, output_dir='output'):
    """绘制性能评估图并一次性保存到output文件夹"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')

    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')

    plt.subplot(1, 3, 3)
    plt.plot(y_true, label='真实值', color='blue', alpha=0.7)
    plt.plot(y_pred, label='预测值', color='orange', alpha=0.7)
    plt.xlabel('样本')
    plt.ylabel('值')
    plt.title('预测值与真实值的折线图')
    plt.legend()

    plt.suptitle(f'{building_name} - 第 {period_index + 1} 个预测周期')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'{building_name}_performance_cycle_{period_index + 1}.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='blue', alpha=0.7)
    plt.xlabel('误差')
    plt.ylabel('频率')
    plt.title(f'{building_name} - 第 {period_index + 1} 个预测周期误差分布')
    plt.savefig(os.path.join(output_dir, f'{building_name}_error_dist_cycle_{period_index + 1}.png'))
    plt.close()

def optimize_weights(X_scaled, y_scaled, weights_range=np.linspace(0, 1, 11)):
    """基于验证集优化N-BEATS和XGBoost的集成权重"""
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    input_shape = (X_scaled.shape[1],)
    nb_beats_model = build_nb_beats_model(input_shape)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    nb_beats_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                       callbacks=[reduce_lr, early_stopping], verbose=1)

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, random_state=42)
    xgb_model.fit(X_train, y_train.ravel())

    y_pred_nb_val_scaled = nb_beats_model.predict(X_val, verbose=0)
    y_pred_xgb_val_scaled = xgb_model.predict(X_val).reshape(-1, 1)

    scaler_y = StandardScaler()
    scaler_y.fit(y_scaled)
    y_pred_nb_val = scaler_y.inverse_transform(y_pred_nb_val_scaled).flatten()
    y_pred_xgb_val = scaler_y.inverse_transform(y_pred_xgb_val_scaled).flatten()
    y_val_true = scaler_y.inverse_transform(y_val).flatten()

    best_weight_nb = 0
    best_score = float('inf')
    for w_nb in weights_range:
        w_xgb = 1 - w_nb
        y_pred_ensemble_val = w_nb * y_pred_nb_val + w_xgb * y_pred_xgb_val
        score = mean_squared_error(y_val_true, y_pred_ensemble_val)
        if score < best_score:
            best_score = score
            best_weight_nb = w_nb

    best_weight_xgb = 1 - best_weight_nb
    print(f"最佳权重: N-BEATS={best_weight_nb:.3f}, XGBoost={best_weight_xgb:.3f}, 验证集MSE={best_score:.4f}")

    nb_beats_model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, validation_split=0.2,
                       callbacks=[reduce_lr, early_stopping], verbose=1)
    xgb_model.fit(X_scaled, y_scaled.ravel())

    return best_weight_nb, best_weight_xgb, nb_beats_model, xgb_model

def perform_temperature_prediction(file_path, weather_data, params):
    try:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"警告: 无法从文件名 {filename} 中提取机组名称")
            return None

        building_name = parts[1]
        is_public_building = building_name in PUBLIC_BUILDINGS
        print(f"正在处理机组: {building_name}, 类型: {'公建' if is_public_building else '非公建'}")

        df = process_data(file_path, params['resample_period'])
        df = create_cyclic_features(df)

        if len(df) <= 24:
            print(f"警告: {file_path} 中的数据点不足，跳过该文件")
            return None

        all_predictions = []
        last_timestamp = df.index[-1]
        segment_hours = int(params['resample_period'][0])

        start_hour = params.get('start_hour', 0)
        for i in range(params['forecast_periods']):
            current_start_hour = (start_hour + i * segment_hours) % 24
            current_start = datetime(2024, 4, 1) + timedelta(hours=current_start_hour)
            current_end = current_start + timedelta(hours=segment_hours)

            if is_public_building:
                X = []
                y = []
                train_end = current_start
                train_start = train_end - timedelta(days=30)
                train_data = df.loc[train_start:train_end]

                for timestamp in train_data.index[:-segment_hours:segment_hours]:
                    period_end = timestamp + timedelta(hours=segment_hours)
                    try:
                        features = prepare_features_for_public_building(
                            train_data, timestamp, period_end, weather_data, params['resample_period']
                        )
                        if not np.any(np.isnan(features)):
                            X.append(features)
                            y.append(train_data.loc[timestamp:period_end]['SST'].mean())
                    except KeyError as e:
                        print(f"警告: 在处理时间戳 {timestamp} 时出现 KeyError: {e}")
                        continue
            else:
                train_end = current_start - timedelta(hours=segment_hours)
                train_start = train_end - timedelta(days=56)
                train_segment = df.loc[train_start:train_end]

                if train_segment.empty:
                    print(
                        f"警告: 时间段 {current_start_hour:02d}:00 - {(current_start_hour + segment_hours) % 24:02d}:00 无有效数据，跳过该时间段")
                    continue

                SST = train_segment['SST'].values
                avg_temp = (train_segment['SST'] + train_segment['SRT']) / 2

                X = []
                y = []

                for j in range(4, len(SST)):
                    features = prepare_features_for_other(
                        train_segment,
                        train_segment.index[j],
                        SST[j - 4:j],
                        avg_temp.iloc[j - 1],
                        weather_data,
                        train_segment.index[j]
                    )
                    if not np.any(np.isnan(features)):
                        X.append(features)
                        y.append(SST[j])

            if not X or not y:
                print(f"警告: 没有足够的训练数据，跳过该时间段")
                continue

            X = np.array(X)
            y = np.array(y).reshape(-1, 1)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            weight_nb, weight_xgb, nb_beats_model, xgb_model = optimize_weights(X_scaled, y_scaled)

            y_pred_nb_scaled = nb_beats_model.predict(X_scaled, verbose=0)
            y_pred_xgb_scaled = xgb_model.predict(X_scaled).reshape(-1, 1)
            y_pred_nb = scaler_y.inverse_transform(y_pred_nb_scaled).flatten()
            y_pred_xgb = scaler_y.inverse_transform(y_pred_xgb_scaled).flatten()
            y_true = scaler_y.inverse_transform(y_scaled).flatten()

            y_pred_ensemble = weight_nb * y_pred_nb + weight_xgb * y_pred_xgb

            evaluate_model(y_true, y_pred_ensemble, "集成模型")
            plot_performance(y_true, y_pred_ensemble, building_name, i, output_dir=params['output_dir'])

            try:
                if is_public_building:
                    prediction_features = prepare_features_for_public_building(
                        df, current_start, current_end, weather_data, params['resample_period']
                    )
                else:
                    last_known = df.loc[:current_start].iloc[-4:]
                    last_SST = last_known['SST'].values
                    last_avg = (last_known['SST'] + last_known['SRT']).mean() / 2
                    prediction_features = prepare_features_for_other(
                        df, current_start, last_SST, last_avg, weather_data, current_end
                    )

                prediction_features_scaled = scaler_X.transform(prediction_features.reshape(1, -1))

                pred_nb_scaled = nb_beats_model.predict(prediction_features_scaled, verbose=0)[0][0]
                pred_nb = scaler_y.inverse_transform([[pred_nb_scaled]])[0][0]

                pred_xgb_scaled = xgb_model.predict(prediction_features_scaled)[0]
                pred_xgb = scaler_y.inverse_transform([[pred_xgb_scaled]])[0][0]

                prediction = weight_nb * pred_nb + weight_xgb * pred_xgb

                prediction = limit_prediction(prediction, params)
                if all_predictions:
                    prediction = limit_change_rate(
                        prediction,
                        all_predictions[-1]['forecastHeat'],
                        params['tempChangeRate']
                    )
                    prediction = limit_prediction(prediction, params)

                all_predictions.append({
                    'projectNo': params['projectNo'],
                    'cim': params['cim'],
                    'timestamp': current_end.strftime('%Y-%m-%d %H:%M:%S'),
                    'forecastHeat': prediction,
                    'forecastCycleType': params['forecastCycleType'],
                    'forecastHeatType': params['forecastHeatType'],
                    'start_hour': current_start_hour,
                    'end_hour': (current_start_hour + segment_hours) % 24
                })

            except Exception as e:
                print(f"警告: 在生成预测时发生错误: {str(e)}")
                continue

        if all_predictions:
            results = pd.DataFrame(all_predictions)
            output_dir = params['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_温度预测值.csv")
            results.to_csv(output_path, index=False)
            print(f"文件 {file_path} 的预测结果已保存")
            return results
        else:
            print(f"警告: 文件 {file_path} 未生成任何预测结果")
            return None

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main(params):
    directory_path = params['input_dir']
    output_dir = params['output_dir']
    weather_data_path = r'E:\PythonProject\weather\Example of hourly weather data.csv'

    weather_data = load_weather_data(weather_data_path)
    all_predictions = []

    for i, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory_path, filename)
            parts = filename.split('_')
            if len(parts) >= 3:
                params['cim'] = f"{parts[1]}_{parts[2]}"
            else:
                print(f"警告: 无法从文件名 {filename} 中提取 cim")
                continue
            result = perform_temperature_prediction(file_path, weather_data, params)
            if result is not None:
                all_predictions.append(result)

    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_output_path = os.path.join(output_dir,
                                            f"combined_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_predictions.to_csv(combined_output_path, index=False)
        print(f"所有预测结果已合并并保存至: {combined_output_path}")
    else:
        print("没有生成任何有效的预测结果。")

if __name__ == "__main__":
    params = {
        'input_dir': r'E:\PythonProject\datasets1',
        'output_dir': r'E:\PythonProject\output',
        'resample_period': '4H',
        'forecast_periods': 6,
        'flowUpperLimit': 80.0,
        'flowLowerLimit': 0.0,
        'projectNo': '1',
        'forecastCycleType': 1,
        'forecastHeatType': 2,
        'start_hour': 3,
        'tempChangeRate': 5.0,
    }

    main(params)