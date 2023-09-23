import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler
import wandb
from IPython.display import Markdown
import datetime


BATCH_SIZE = 10
FEATURES = ['timestamp', 
    'INV01_CurrentDC(A)', 'INV01_CurrentAC(A)', 'INV01_TotalEnergy(kWh)','INV01_PowerAC(kW)', 'INV01_PowerDC(kW)',
       'INV01_InternalTemperature(C)', 'INV01_HeatSinkTemperature(C)',
       'INV01_VoltageDC(V)', 'INV01_VoltageAC(V)', 'INV02_CurrentDC(A)',
       'INV02_CurrentAC(A)', 'INV02_TotalEnergy(kWh)', 'INV02_PowerAC(kW)',
       'INV02_PowerDC(kW)', 'INV02_InternalTemperature(C)',
       'INV02_HeatSinkTemperature(C)', 'INV02_VoltageDC(V)',
       'INV02_VoltageAC(V)', 'INV03_CurrentDC(A)', 'INV03_CurrentAC(A)',
       'INV03_TotalEnergy(kWh)', 'INV03_PowerAC(kW)', 'INV03_PowerDC(kW)',
       'INV03_InternalTemperature(C)', 'INV03_HeatSinkTemperature(C)',
       'INV03_VoltageDC(V)', 'INV03_VoltageAC(V)', 'Cont_TotalEnergy(kWh)', 'Cont_TotalEnergyImported(kWh)',
       'Impianto_SolargisGHI(W/m2)', 'Impianto_SolargisGTI(W/m2)',
       'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
       'temperature_2m (°C)', 'relativehumidity_2m (%)', 'dewpoint_2m (°C)', 'rain (mm)', 'cloudcover (%)',
       'soil_temperature_7_to_28cm (°C)', 'soil_moisture_7_to_28cm (m³/m³)'
]


def preprocess_data(dataset_path: str, isday_path: str):
    df = pd.read_csv(dataset_path, delimiter = ";")
    df['timestamp'] = pd.to_datetime(df['timestamp'], yearfirst=True)
    df = df[FEATURES]
    
    # aggiunta encoding ciclico minuti, giorni
    tmp_datetime_serie = pd.to_datetime(arg=df['timestamp'], yearfirst=True)

    df["minute_sin"] = np.sin(2 * np.pi * tmp_datetime_serie.dt.minute / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * tmp_datetime_serie.dt.minute / 60.0)

    df["day_sin"] = np.sin(2 * np.pi * tmp_datetime_serie.dt.day / 31.0)
    df["day_cos"] = np.cos(2 * np.pi * tmp_datetime_serie.dt.day / 31.0)

    # trasformo la target series in Non Cumulativa
    df['target'] = df['Cont_TotalEnergy(kWh)'].diff().fillna(0)
    
    df = df.set_index(df['timestamp'])
    
    media_prod_giornaliera = __media_giornaliera(df)
    
    df = df.resample('15Min').sum() # No sum su TotalEnergy
    
    # Custom ISDay
    df = __custom_isday(isday_path, df)
    
    # Rimozione giorni mancanti
    df = __remove_missing_days(df)
    
    # Train, Val, Test split
    datasets = __dataset_splitting(df)
    
    return {"datasets": datasets, "media": media_prod_giornaliera, "df": df,}
    

def __media_giornaliera(df: pd.DataFrame):
    media_prod_giornaliera = df.resample('1d').sum()['target'].mean()
    
    return media_prod_giornaliera


def __custom_isday(isday_path: str, df: pd.DataFrame):
    
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    
    def custom_isday(sunrise, sunset):
        tmp_df = pd.DataFrame(
            0,
            index=df[
                ((df.index >= pd.to_datetime(sunrise.date())) * 
                 (df.index < pd.to_datetime(sunrise.date() + datetime.timedelta(days=1))))
            ].index,
            columns=['isday']
        )

        # print(tmp_df.shape)

        start = nearest(tmp_df.index, sunrise)
        end = nearest(tmp_df.index,sunset)

        # print(start, "-", end)

        tmp_df[((tmp_df.index >= start) * (tmp_df.index <= end))] = 1

        return tmp_df

    sunsetrise = pd.read_csv(isday_path, header=2, sep=',', index_col=False, parse_dates=[
    'time', 'sunrise (iso8601)', 'sunset (iso8601)'])
    sunsetrise.index = sunsetrise['time']
    sunsetrise.drop(columns=['time'], inplace=True)
    sunsetrise.columns = ['sunrise', 'sunset']
    sunsetrise = sunsetrise.loc[:df.index[-1]]
    
    customisday = pd.DataFrame()

    for index, (sunrise, sunset) in sunsetrise.iterrows():
        test = custom_isday(sunrise, sunset)
        customisday = pd.concat([customisday, test])
        
    df = pd.concat([df, customisday], axis=1)
    
    return df


def __remove_missing_days(df: pd.DataFrame):
    
    def remove_day(day):
        to_drop = df[(((df.index >= pd.to_datetime(day)) * (df.index < (pd.to_datetime(day) + datetime.timedelta(days=1)))))].index

        df.drop(to_drop, inplace=True)

    DATES_TO_REMOVE = [
        '2022-06-09', '2022-06-10', '2022-06-11', '2022-06-12', '2022-06-13',
           '2022-06-28', '2022-06-29', '2022-06-30', '2022-08-26', '2022-09-23',
           '2022-10-06', '2023-02-03', '2023-02-15', '2023-02-16', '2023-03-26',
    ]
    
    for day in DATES_TO_REMOVE:
        remove_day(day)
        
    return df
    
    
def __dataset_splitting(df: pd.DataFrame):
    # original
    train_df = df[((df.index >= '2022-06-01') * (df.index < '2023-01-01'))]
    val_df = df[((df.index >= '2023-01-01') * (df.index < '2023-02-01'))]
    test_df = df[((df.index >= '2023-02-01') * (df.index < '2023-03-01'))]
    
    train_scaler = MinMaxScaler()
    train_scaler.fit(train_df)

    train_target_scaler = MinMaxScaler()
    train_target_scaler.fit(train_df['target'].to_numpy().reshape(-1, 1))

    train_df = pd.DataFrame(train_scaler.transform(train_df), columns=train_df.columns, index=train_df.index)
    val_df = pd.DataFrame(train_scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    test_df = pd.DataFrame(train_scaler.transform(test_df), columns=test_df.columns, index=test_df.index)
    
    # test 1
    test_df_1 = df[((df.index >= '2023-03-01') * (df.index < '2023-04-01'))]
    test_df_2 = df[((df.index >= '2023-04-01') * (df.index < '2023-05-01'))]
    
    test_df_1 = pd.DataFrame(train_scaler.transform(test_df_1), columns=test_df_1.columns, index=test_df_1.index)
    test_df_2 = pd.DataFrame(train_scaler.transform(test_df_2), columns=test_df_2.columns, index=test_df_2.index)
    
    # test 2
    train_df_2 = df[((df.index >= '2022-06-01') * (df.index < '2023-03-01'))]
    val_df_2 = df[((df.index >= '2023-03-01') * (df.index < '2023-04-01'))]
    test_df_2_2 = df[((df.index >= '2023-04-01') * (df.index < '2023-05-01'))]
    
    train_scaler_2 = MinMaxScaler()
    train_scaler_2.fit(train_df_2)

    train_target_scaler_2 = MinMaxScaler()
    train_target_scaler_2.fit(train_df_2['target'].to_numpy().reshape(-1, 1))

    train_df_2 = pd.DataFrame(train_scaler_2.transform(train_df_2), columns=train_df_2.columns, index=train_df_2.index)
    val_df_2 = pd.DataFrame(train_scaler_2.transform(val_df_2), columns=val_df_2.columns, index=val_df_2.index)
    test_df_2_2 = pd.DataFrame(train_scaler_2.transform(test_df_2_2), columns=test_df_2_2.columns, index=test_df_2_2.index)
    
    return {
        "original": [train_df, val_df, test_df, train_scaler, train_target_scaler],
        "test 1": [train_df, val_df, test_df_1, test_df_2, train_scaler, train_target_scaler],
        "test 2": [train_df_2, val_df_2, test_df_2_2, train_scaler_2, train_target_scaler_2]
    }


class CustomDataset(Dataset):
    MAX_DAYS = 4
    MIN_DAYS = 1
    DAY_IN_TIMESTAMPS = 96 # 15 min

    def __init__(self, df, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.counter = 0
        self.target_len = -1
    
    def __len__(self):
        return int(len(self.df)/self.DAY_IN_TIMESTAMPS)-2 - self.MAX_DAYS
    
    def __getitem__(self, idx):
        if self.counter == self.batch_size:
            self.counter = 0
        
        if self.counter == 0:
            self.target_len = np.random.randint(1, self.MAX_DAYS+1)
        
        tmp_df = self.df.iloc[0+(self.DAY_IN_TIMESTAMPS*(idx+1)) - self.DAY_IN_TIMESTAMPS : (self.DAY_IN_TIMESTAMPS*(idx+1))+(self.DAY_IN_TIMESTAMPS*self.target_len) + self.DAY_IN_TIMESTAMPS]
        timestamps = tmp_df.index

        mask_before = np.array([1]*self.DAY_IN_TIMESTAMPS + [0]*(len(tmp_df) - self.DAY_IN_TIMESTAMPS))
        mask_after = np.array([0] * (len(tmp_df) - self.DAY_IN_TIMESTAMPS) + [1]*self.DAY_IN_TIMESTAMPS)
        mask_target = np.array([0]*self.DAY_IN_TIMESTAMPS + [1] * (len(tmp_df) - self.DAY_IN_TIMESTAMPS*2) +[0] * self.DAY_IN_TIMESTAMPS)
                
        before = tmp_df[mask_before == 1]
        target = tmp_df[mask_target == 1]
        after  = tmp_df[mask_after == 1]
        
        future = target.copy()
        future = future[[
            'Impianto_SolargisGHI(W/m2)','Impianto_SolargisGTI(W/m2)', 
            'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
            'temperature_2m (°C)', 'relativehumidity_2m (%)', 'dewpoint_2m (°C)', 
            'rain (mm)', 'cloudcover (%)', 'soil_temperature_7_to_28cm (°C)', 'soil_moisture_7_to_28cm (m³/m³)',
            'isday',
            #'is_day ()',
        ]]
        
        tsolargis = target[['Impianto_SolargisGHI(W/m2)','Impianto_SolargisGTI(W/m2)']].copy()
        tsolargis.columns = ['GHI', 'GTI']
        
        target = target['target']
        
        self.counter += 1
        
        return before, target, future, after, mask_before, mask_target, mask_after, timestamps, tsolargis
    
    def reset(self):
        self.counter = 0
        
def collate_fn(batch):
    # ha senso usare del padding ??
    before, target, future, after, masks_before, masks_target, masks_after, timestamps, solargis = zip(*batch)
    
    before = [torch.tensor(d.values, dtype=torch.float32) for d in before]
    before = torch.stack(before)
    
    target = [torch.tensor(d.values, dtype=torch.float32) for d in target]
    target = [torch.reshape(d, (d.shape[0], 1)) for d in target] # voglio shape [len_buco, 1]
    target = torch.stack(target)
    
    future = [torch.tensor(d.values, dtype=torch.float32) for d in future]
    future = torch.stack(future)
    
    after = [torch.tensor(d.values, dtype=torch.float32) for d in after]
    after = torch.stack(after)
    
    return before, target, future, after, masks_before, masks_target, masks_after, timestamps, solargis