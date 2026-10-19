import numpy as np
import pandas as pd
from scipy.signal import welch

# pasma częstotliwosci
EEG_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta':  (12, 30)
}

def get_band_power(signal, sfreq, band):
    # obliczamy za pomocą metody welcha bezwzględną moc sygnału,
    # dla sreq = 128 hz, z precyzją 1 hz
    freqs, psd = welch(signal, fs=sfreq, nperseg=sfreq)
    # bierzemy tylko indexy interesującego nas pasma
    band_idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    # zwracamy srednią ze wszystkich sygnałów z danego okna
    return np.mean(psd[band_idx])

# ustawiamy sfreq=128, jest to częstotliwosć próbkowania, która w naszym zbiorze wynosi 128hz
def extract_features_for_subject(subject_df, channels, sfreq=128):
    features = {}
    
    for channel in channels:
        signal = subject_df[channel].values
        
        # ekstrakcja mocy pasm
        for band_name, band_range in EEG_BANDS.items():
            power = get_band_power(signal, sfreq, band_range)
            features[f'{channel}_{band_name}'] = power
            
        # obliczamy theta / beta ratio
        features[f'{channel}_TBR'] = features[f'{channel}_Theta'] / (features[f'{channel}_Beta'] + 1e-6)
        
    # obliczamy znormalizowany wskaźnik asymetrii czołowej, frontal assymetry
    f4_alpha = features['F4_Alpha']
    f3_alpha = features['F3_Alpha']
    features['FAI_Alpha_F4_F3'] = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-6)

    return pd.Series(features)

def build_dataset(df, window_sec=2, target_col='Class', group_col='ID', sfreq=128):
    channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'O1', 'O2']
    window_samples = window_sec * sfreq
    feature_list = []
    
    grouped = df.groupby(group_col)
    
    for subject_id, group in grouped:
        n_samples = len(group)
        # obliczamy ile okien możemy wyciągnąć dla danego pacjenta
        n_windows = n_samples // window_samples
        
        for i in range(n_windows):
            # bierzemy fragment dla danego okna
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            window_df = group.iloc[start_idx:end_idx]
            
            subj_features = extract_features_for_subject(window_df, channels, sfreq)
            
            # dodajemy resztę danych, w tyn numer okna
            subj_features[group_col] = subject_id
            subj_features['window'] = i
            subj_features[target_col] = group[target_col].iloc[0] 
            
            feature_list.append(subj_features)
            
    return pd.DataFrame(feature_list)