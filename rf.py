# '''
# Version 5.8 (Deeper Models & Enhanced Output)
# - Model architectures for LSTM, MLP, and CNN have been made deeper and more
#   complex by stacking layers and increasing units/filters.
# - Dropout and BatchNormalization layers have been added to prevent overfitting.
# - The final output CSV now includes the original input features and true target
#   values alongside the predictions for a more comprehensive analysis.
# - Predicted chain values are now saved in separate columns (pred_num_c_1, etc.)
#   for easier comparison.
# '''

import pandas as pd
import numpy as np
import ast
import os
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
from collections import Counter
import warnings
import joblib

# --- New Imports for Deep Learning Model ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Attention, Concatenate, GaussianNoise, Conv1D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# Set a seed for reproducibility of the deep learning model
tf.random.set_seed(42)
np.random.seed(42)


# --- 0. Configuration ---
# Path to your data files
try:
    # Create dummy files if they don't exist to prevent errors on first run
    TRAINING_FILE = '/Users/bowen/Desktop/DeepLipid/peak_formula_annotation/output/train_new.csv'
    TEST_FILE_1 = '/Users/bowen/Desktop/DeepLipid/peak_formula_annotation/output/test_new.csv'
    TEST_FILE_2 = '/Users/bowen/Desktop/DeepLipid/datasets/Li_Lab_Lipid_Stanard/standard_df.csv'

    if not os.path.exists(TRAINING_FILE): pd.DataFrame().to_csv(TRAINING_FILE)
    if not os.path.exists(TEST_FILE_1): pd.DataFrame().to_csv(TEST_FILE_1)
    if not os.path.exists(TEST_FILE_2): pd.DataFrame().to_csv(TEST_FILE_2)
except Exception as e:
    print(f"Could not create dummy files. Please ensure data files exist. Error: {e}")
    TRAINING_FILE = '/Users/bowen/Desktop/DeepLipid/peak_formula_annotation/output/train_new.csv'
    TEST_FILE_1 = '/Users/bowen/Desktop/DeepLipid/peak_formula_annotation/output/test_new.csv'
    TEST_FILE_2 = '/Users/bowen/Desktop/DeepLipid/datasets/Li_Lab_Lipid_Stanard/standard_df.csv'


# MS2 Processing Parameters
MS2_FEATURE_DIM = 500
MS2_MZ_MIN = 10
MS2_MZ_MAX = 2000
MS2_MZ_STEP = 0.1
MS2_INTENSITY_THRESHOLD = 5.0

# Target variable definitions
STAGE_1_TARGETS = ['num_c', 'num_db']
STAGE_2_TARGETS = ['num_c_1', 'num_db_1', 'num_c_2', 'num_db_2', 'num_c_3', 'num_db_3', 'num_c_4', 'num_db_4']
ALL_TARGETS = STAGE_1_TARGETS + STAGE_2_TARGETS
NUM_SEQ_OUTPUTS = len(STAGE_2_TARGETS)

# Model & Data Split Parameters
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
DL_EPOCHS = 1
DL_BATCH_SIZE = 8
NOISE_LEVEL = 0.1
MAX_REPREDICTION_TRIES = 5
MODEL_TYPES_TO_TEST = ['LSTM', 'MLP', 'CNN'] # Models to test


# Output folder setup
current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FOLDER = f"results_{current_time_str}_v5.8_DeeperModels"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Configuration loaded. Output will be saved to: {OUTPUT_FOLDER}")

# --- Global storage ---
label_encoders_dict = {}
scalers = {}
ms2_feature_columns = [f"ms2_mz_{mz:.1f}" for mz in np.arange(MS2_MZ_MIN, MS2_MZ_MAX, MS2_MZ_STEP)]

# --- 1. Helper & Preprocessing Functions ---

def parse_formula(formula_str):
    if not isinstance(formula_str, str):
        return pd.Series({'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0, 'P': 0})
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    counts = Counter()
    for element, count in elements:
        counts[element] += int(count) if count else 1
    return pd.Series({
        'formula_C': counts.get('C', 0), 'formula_H': counts.get('H', 0),
        'formula_O': counts.get('O', 0), 'formula_N': counts.get('N', 0),
        'formula_S': counts.get('S', 0), 'formula_P': counts.get('P', 0)
    })

def process_ms2_spectra(df):
    ms2_df = pd.DataFrame(0, index=df.index, columns=ms2_feature_columns)
    if 'MS2_norm' not in df.columns: return ms2_df
    for idx, row in df.iterrows():
        try:
            ms2_data = ast.literal_eval(row['MS2_norm']) if isinstance(row['MS2_norm'], str) else row['MS2_norm']
            if not isinstance(ms2_data, list): continue
            for mz, intensity in ms2_data:
                if MS2_MZ_MIN <= mz < MS2_MZ_MAX:
                    final_intensity = intensity if intensity >= MS2_INTENSITY_THRESHOLD else 0.0
                    if final_intensity > 0:
                        col_name = f"ms2_mz_{round(mz, 1):.1f}"
                        if col_name in ms2_df.columns:
                            ms2_df.loc[idx, col_name] = final_intensity
        except (ValueError, SyntaxError, TypeError): continue
    return ms2_df

def preprocess_base_features(df_input, is_train_phase):
    global label_encoders_dict
    df = df_input.copy()
    base_features = ['precursor_mz', 'num_peaks', 'num_chain']
    for col in base_features:
        if col not in df.columns: df[col] = 0
    
    if 'formula' not in df.columns: df['formula'] = ""
    formula_feats = df['formula'].apply(parse_formula)

    categorical_cols = ['class', 'adduct']
    encoded_cats = pd.DataFrame(index=df.index)
    for col in categorical_cols:
        if col not in df.columns: df[col] = 'unknown'
        df[col] = df[col].astype(str)
        if is_train_phase:
            encoder = LabelEncoder()
            encoded_cats[col + '_encoded'] = encoder.fit_transform(df[col])
            label_encoders_dict[col] = encoder
        else:
            encoder = label_encoders_dict.get(col)
            if encoder:
                encoded_cats[col + '_encoded'] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            else:
                encoded_cats[col + '_encoded'] = -1

    X_base = pd.concat([df[base_features], formula_feats, encoded_cats], axis=1)
    return X_base.fillna(0)

def canonicalize_chains(row_values):
    if not isinstance(row_values, np.ndarray): row_values = np.array(row_values)
    chains = row_values.reshape(-1, 2)
    active_chains = chains[chains[:, 0] > 0]
    sorted_chains = sorted(active_chains.tolist(), key=lambda x: (-x[0], -x[1]))
    padded_sorted_chains = np.zeros((4, 2), dtype=int)
    if sorted_chains:
        padded_sorted_chains[:len(sorted_chains), :] = sorted_chains
    return padded_sorted_chains.flatten()

def adjust_sum(predictions, target_sum):
    predictions = np.maximum(0, predictions).astype(float)
    if target_sum == 0: return np.zeros_like(predictions, dtype=int)
    current_sum = np.sum(predictions)
    if current_sum == 0: return np.zeros_like(predictions, dtype=int)
    scaled = (predictions / current_sum) * target_sum
    int_parts = np.floor(scaled).astype(int)
    remainders = scaled - int_parts
    missing = int(round(target_sum - np.sum(int_parts)))
    if missing > 0:
        dist_indices = np.argsort(remainders)[-missing:]
        int_parts[dist_indices] += 1
    return int_parts

# --- New Single-Chain Calculation Functions ---
def _calculate_mass_from_formula(formula: str, atom_masses: dict) -> float:
    """Calculates the monoisotopic mass from a chemical formula string."""
    total_mass = 0.0
    pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    matches = pattern.findall(formula)
    if not matches: raise ValueError(f"Could not parse the formula: '{formula}'")
    for element, count_str in matches:
        if element not in atom_masses: raise ValueError(f"Unknown element '{element}' in formula.")
        count = int(count_str) if count_str else 1
        total_mass += atom_masses[element] * count
    return total_mass

def find_lipid_composition(formula: str, lipid_class: str, tolerance_ppm: float = 30.0):
    """Calculates possible C:DB counts for a lipid's fatty acyl/alkyl tail."""
    ATOM_MASSES = {'C': 12.0, 'H': 1.007825, 'O': 15.994915, 'N': 14.003074, 'P': 30.973762}
    MASS_H2 = 2 * ATOM_MASSES['H']
    MASS_O2 = 2 * ATOM_MASSES['O']
    HEAD_GROUP_MASSES = {
        'FA': 0.0, 'MG': 74.03678, 'LPA': 154.003112, 'LPC': 239.092261,
        'LPE': 197.045311, 'LPG': 228.039892, 'LPI': 316.055937, 'LPS': 241.035141,
        'CAR': 143.094629, 'NAE': 43.042199, 'CE': 368.344350, 'ST': 368.344350,
        'LDGTS': 217.131409, 'LDGCC': 398.142430, 'LPC-O': 239.092261, 'LPE-O': 197.045311
    }
    ETHER_CLASSES = {'LPC-O', 'LPE-O'}

    if lipid_class not in HEAD_GROUP_MASSES:
        return []

    try:
        exact_mass = _calculate_mass_from_formula(formula, ATOM_MASSES)
    except ValueError:
        return []
        
    tail_mass = exact_mass - HEAD_GROUP_MASSES[lipid_class]
    if tail_mass <= 0: return []

    possible_matches = []
    is_ether = lipid_class in ETHER_CLASSES
    for n_carbons in range(1, 41):
        if is_ether:
            saturated_mass = (n_carbons * (ATOM_MASSES['C'] + MASS_H2)) + MASS_H2 + ATOM_MASSES['O']
        else:
            saturated_mass = (n_carbons * (ATOM_MASSES['C'] + MASS_H2)) + MASS_O2
        for n_double_bonds in range(0, 11):
            theoretical_mass = saturated_mass - (n_double_bonds * MASS_H2)
            ppm_error = (abs(tail_mass - theoretical_mass) / tail_mass) * 1_000_000
            if ppm_error <= tolerance_ppm:
                possible_matches.append([n_carbons, n_double_bonds])
    return possible_matches

def process_single_chain_lipids(df_single_chain):
    """Calculates predictions for single-chain lipids using the formula-based method."""
    predictions = []
    for _, row in df_single_chain.iterrows():
        formula = row.get('formula', '')
        lipid_class = row.get('class', '')
        
        if not isinstance(formula, str):
            formula = ''

        c, db = 0, 0
        matches = find_lipid_composition(formula, lipid_class)
        if matches: c, db = matches[0]
        
        pred_s2_array = np.zeros(len(STAGE_2_TARGETS), dtype=int)
        pred_s2_array[0], pred_s2_array[1] = c, db
        predictions.append(pred_s2_array)

    results_df = df_single_chain.copy()
    pred_cols = [f'pred_{t}' for t in STAGE_2_TARGETS]
    pred_df = pd.DataFrame(predictions, columns=pred_cols, index=results_df.index)
    results_df = pd.concat([results_df, pred_df], axis=1)
    results_df['is_valid_prediction'] = True
    return results_df

# --- 2. Modeling & Evaluation Functions (Deeper Architectures) ---

def build_stage2_lstm_model(num_base_s1_features, num_ms2_features):
    """Builds a deeper LSTM model with stacked layers and dropout."""
    input_base_s1 = Input(shape=(num_base_s1_features,), name='input_base_s1')
    input_ms2 = Input(shape=(num_ms2_features,), name='input_ms2')
    
    noisy_ms2 = GaussianNoise(NOISE_LEVEL, name='gaussian_noise')(input_ms2)
    concatenated_features = Concatenate(name='concatenate_features')([input_base_s1, noisy_ms2])
    
    repeated_features = RepeatVector(NUM_SEQ_OUTPUTS, name='repeat_vector')(concatenated_features)
    
    # Deeper LSTM structure
    lstm_1 = LSTM(256, return_sequences=True, name='lstm_layer_1')(repeated_features)
    dropout_1 = Dropout(0.3, name='dropout_1')(lstm_1)
    lstm_2 = LSTM(128, return_sequences=True, name='lstm_layer_2')(dropout_1)
    dropout_2 = Dropout(0.3, name='dropout_2')(lstm_2)
    
    attention_out = Attention(name='attention_layer')([dropout_2, dropout_2])
    
    # Additional dense layer for more complex transformations
    dense_out = TimeDistributed(Dense(64, activation='relu'), name='dense_transform')(attention_out)
    output = TimeDistributed(Dense(1, activation='relu'), name='output_layer')(dense_out)
    
    model = Model(inputs=[input_base_s1, input_ms2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_stage2_mlp_model(num_base_s1_features, num_ms2_features):
    """Builds a deeper MLP-style model with multiple dense layers."""
    input_base_s1 = Input(shape=(num_base_s1_features,), name='input_base_s1')
    input_ms2 = Input(shape=(num_ms2_features,), name='input_ms2')

    noisy_ms2 = GaussianNoise(NOISE_LEVEL, name='gaussian_noise')(input_ms2)
    concatenated_features = Concatenate(name='concatenate_features')([input_base_s1, noisy_ms2])

    # Deeper MLP structure
    dense_1 = Dense(256, activation='relu', name='dense_1')(concatenated_features)
    bn_1 = BatchNormalization(name='bn_1')(dense_1)
    dropout_1 = Dropout(0.4, name='dropout_1')(bn_1)
    
    dense_2 = Dense(128, activation='relu', name='dense_2')(dropout_1)
    bn_2 = BatchNormalization(name='bn_2')(dense_2)
    dropout_2 = Dropout(0.4, name='dropout_2')(bn_2)

    repeated_features = RepeatVector(NUM_SEQ_OUTPUTS, name='repeat_vector')(dropout_2)
    attention_out = Attention(name='attention_layer')([repeated_features, repeated_features])
    
    output = TimeDistributed(Dense(1, activation='relu'), name='output_layer')(attention_out)
    
    model = Model(inputs=[input_base_s1, input_ms2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_stage2_cnn_model(num_base_s1_features, num_ms2_features):
    """Builds a deeper CNN model with stacked convolutional layers."""
    input_base_s1 = Input(shape=(num_base_s1_features,), name='input_base_s1')
    input_ms2 = Input(shape=(num_ms2_features,), name='input_ms2')

    noisy_ms2 = GaussianNoise(NOISE_LEVEL, name='gaussian_noise')(input_ms2)
    concatenated_features = Concatenate(name='concatenate_features')([input_base_s1, noisy_ms2])
    
    repeated_features = RepeatVector(NUM_SEQ_OUTPUTS, name='repeat_vector')(concatenated_features)
    
    # Deeper CNN structure
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv1d_1')(repeated_features)
    bn_1 = BatchNormalization(name='bn_1')(conv1)
    dropout_1 = Dropout(0.3, name='dropout_1')(bn_1)

    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv1d_2')(dropout_1)
    bn_2 = BatchNormalization(name='bn_2')(conv2)
    dropout_2 = Dropout(0.3, name='dropout_2')(bn_2)

    attention_out = Attention(name='attention_layer')([dropout_2, dropout_2])
    output = TimeDistributed(Dense(1, activation='relu'), name='output_layer')(attention_out)
    
    model = Model(inputs=[input_base_s1, input_ms2], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_pipeline(X_train_base, Y_train, df_train_full, model_type='LSTM'):
    global scalers
    print("  1. Training Stage 1 model (num_c, num_db)...")
    X_train_s1 = X_train_base.copy()
    model_s1 = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model_s1.fit(X_train_s1, Y_train[STAGE_1_TARGETS])

    print("  2. Processing and reducing MS2 data for Stage 2...")
    X_train_ms2_raw = process_ms2_spectra(df_train_full)
    reducer = SelectKBest(f_regression, k=MS2_FEATURE_DIM)
    y_for_f_reg = Y_train['num_c']
    X_train_ms2_reduced_np = reducer.fit_transform(X_train_ms2_raw, y_for_f_reg)
    
    print("  3. Preparing data for Stage 2 Sequential Model...")
    pred_s1_train = np.round(model_s1.predict(X_train_s1)).astype(int)
    
    X_train_s2_base_s1 = np.concatenate([X_train_s1.values, pred_s1_train], axis=1)
    X_train_s2_ms2 = X_train_ms2_reduced_np

    scaler_base_s1 = StandardScaler()
    scaler_ms2 = StandardScaler()
    X_train_s2_base_s1_scaled = scaler_base_s1.fit_transform(X_train_s2_base_s1)
    X_train_s2_ms2_scaled = scaler_ms2.fit_transform(X_train_s2_ms2)
    scalers = {'s2_base_s1': scaler_base_s1, 's2_ms2': scaler_ms2}

    Y_train_s2_reshaped = Y_train[STAGE_2_TARGETS].values.reshape(-1, NUM_SEQ_OUTPUTS, 1)

    print(f"  4. Building and Training Stage 2 {model_type} model...")
    if model_type == 'LSTM':
        model_builder = build_stage2_lstm_model
    elif model_type == 'MLP':
        model_builder = build_stage2_mlp_model
    elif model_type == 'CNN':
        model_builder = build_stage2_cnn_model
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model_s2 = model_builder(
        num_base_s1_features=X_train_s2_base_s1_scaled.shape[1],
        num_ms2_features=X_train_s2_ms2_scaled.shape[1]
    )
    model_s2.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model_s2.fit(
        {'input_base_s1': X_train_s2_base_s1_scaled, 'input_ms2': X_train_s2_ms2_scaled},
        Y_train_s2_reshaped,
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    models = {'s1': model_s1, 's2': model_s2}
    feature_columns = {'s1': X_train_s1.columns}
    
    return models, reducer, feature_columns, scalers

def evaluate_pipeline(df_test_orig, models, reducer, feature_columns, current_scalers):
    # Separate single-chain lipids for direct calculation
    df_single_chain = df_test_orig[df_test_orig['num_chain'] == 1].copy()
    df_multi_chain = df_test_orig[df_test_orig['num_chain'] != 1].copy()

    # Process single-chain lipids
    single_chain_results_df = pd.DataFrame()
    if not df_single_chain.empty:
        print(f"    Directly calculating {len(df_single_chain)} single-chain lipids...")
        single_chain_results_df = process_single_chain_lipids(df_single_chain)

    # Process multi-chain lipids with ML model
    multi_chain_results_df = pd.DataFrame()
    if not df_multi_chain.empty:
        print(f"    Predicting {len(df_multi_chain)} multi-chain lipids with ML model...")
        X_test_base = preprocess_base_features(df_multi_chain, is_train_phase=False)
        X_test_base = X_test_base.reindex(columns=feature_columns['s1'], fill_value=0)
        pred_test_s1 = np.round(models['s1'].predict(X_test_base)).astype(int)

        X_test_ms2_raw = process_ms2_spectra(df_multi_chain)
        X_test_ms2_reduced_np = reducer.transform(X_test_ms2_raw)
        
        X_test_s2_base_s1 = np.concatenate([X_test_base.values, pred_test_s1], axis=1)
        X_test_s2_ms2 = X_test_ms2_reduced_np

        X_test_s2_base_s1_scaled = current_scalers['s2_base_s1'].transform(X_test_s2_base_s1)
        X_test_s2_ms2_scaled = current_scalers['s2_ms2'].transform(X_test_s2_ms2)
        
        pred_test_s2_raw_reshaped = models['s2'].predict(
            {'input_base_s1': X_test_s2_base_s1_scaled, 'input_ms2': X_test_s2_ms2_scaled}
        )
        pred_test_s2_raw = pred_test_s2_raw_reshaped.squeeze(axis=-1)

        predictions_final, validity_checks = [], []
        for i in range(len(pred_test_s2_raw)):
            final_pred_for_sample = None
            is_valid_for_sample = False
            last_attempted_pred = None

            for attempt in range(MAX_REPREDICTION_TRIES):
                current_raw_pred = pred_test_s2_raw[i].copy()
                if attempt > 0:
                    noise = np.random.normal(0, 0.5, current_raw_pred.shape)
                    current_raw_pred += noise

                pred_row_adj = np.round(current_raw_pred).astype(int)
                num_chain_ref = df_multi_chain.iloc[i]['num_chain']
                num_c_ref, num_db_ref = pred_test_s1[i, 0], pred_test_s1[i, 1]
                
                c_indices = [j for j, col in enumerate(STAGE_2_TARGETS) if '_c_' in col]
                db_indices = [j for j, col in enumerate(STAGE_2_TARGETS) if '_db_' in col]
                raw_c, raw_db = pred_row_adj[c_indices], pred_row_adj[db_indices]
                
                indices_to_consider = np.argsort(raw_c)[-int(num_chain_ref):] if num_chain_ref > 0 else []
                
                final_c, final_db = np.zeros_like(raw_c), np.zeros_like(raw_db)
                if num_chain_ref > 0:
                    corrected_c = adjust_sum(raw_c[indices_to_consider], num_c_ref)
                    corrected_db = adjust_sum(raw_db[indices_to_consider], num_db_ref)
                    np.put(final_c, indices_to_consider, corrected_c)
                    np.put(final_db, indices_to_consider, corrected_db)
                
                final_pred = np.zeros_like(pred_row_adj)
                for j, idx in enumerate(c_indices): final_pred[idx] = final_c[j]
                for j, idx in enumerate(db_indices): final_pred[idx] = final_db[j]
                last_attempted_pred = final_pred.copy()

                is_valid = True
                chains = final_pred.reshape(-1, 2)
                if np.count_nonzero(chains[:, 0]) != num_chain_ref: is_valid = False
                if chains[:, 0].sum() != num_c_ref: is_valid = False
                if chains[:, 1].sum() != num_db_ref: is_valid = False
                for c, db in chains:
                    if c > 0 and not (c < 100 and db < 40 and db < c):
                        is_valid = False
                
                if is_valid:
                    final_pred_for_sample = final_pred
                    is_valid_for_sample = True
                    break
            
            if not is_valid_for_sample:
                final_pred_for_sample = last_attempted_pred
                is_valid_for_sample = False

            predictions_final.append(final_pred_for_sample)
            validity_checks.append(is_valid_for_sample)

        multi_chain_results_df = df_multi_chain.copy()
        pred_cols = [f'pred_{t}' for t in STAGE_2_TARGETS]
        pred_df = pd.DataFrame(predictions_final, columns=pred_cols, index=multi_chain_results_df.index)
        multi_chain_results_df = pd.concat([multi_chain_results_df, pred_df], axis=1)
        multi_chain_results_df['is_valid_prediction'] = validity_checks

    # Combine results and calculate metrics
    df_pred_out = pd.concat([single_chain_results_df, multi_chain_results_df]).sort_index()
    
    # Calculate accuracy based on canonical representation
    Y_test_s2 = df_test_orig.loc[df_pred_out.index][STAGE_2_TARGETS]
    true_canonical = [canonicalize_chains(row).tobytes() for _, row in Y_test_s2.iterrows()]
    
    pred_cols = [f'pred_{t}' for t in STAGE_2_TARGETS]
    pred_canonical = [canonicalize_chains(row.values).tobytes() for _, row in df_pred_out[pred_cols].iterrows()]
    
    accuracy = accuracy_score(true_canonical, pred_canonical)
    validation_rate = df_pred_out['is_valid_prediction'].mean()

    return accuracy, validation_rate, df_pred_out

# --- 3. Load and Prepare Data ---
print("\n--- Loading and Preparing Data ---")
all_test_dfs = {}
try:
    df_train_orig = pd.read_csv(TRAINING_FILE)
    all_test_dfs["Test_Set_1"] = pd.read_csv(TEST_FILE_1)
    all_test_dfs["Test_Set_2"] = pd.read_csv(TEST_FILE_2)
except FileNotFoundError as e:
    print(f"\nERROR: Could not read data file: {e}. Exiting.")
    exit()

df_train, df_eval = train_test_split(df_train_orig, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)
all_test_dfs["Evaluation_Set"] = df_eval

for name, df in all_test_dfs.items():
    for col in ['num_chain'] + ALL_TARGETS:
        if col not in df.columns: df[col] = 0

# --- 4. Run Pipeline for All Model Types ---
all_artifacts = {}

for model_type in MODEL_TYPES_TO_TEST:
    print(f"\n\n{'='*25} TESTING MODEL TYPE: {model_type.upper()} {'='*25}")
    
    # Exclude single-chain lipids from training
    df_train_current = df_train[df_train['num_chain'] != 1].copy().reset_index(drop=True)
    print(f"Training on {len(df_train_current)} multi-chain lipids.")
    
    X_train_base = preprocess_base_features(df_train_current, is_train_phase=True)
    Y_train = df_train_current[ALL_TARGETS]

    print(f"\n--- Running Training Pipeline for {model_type} ---")
    models, reducer, feature_cols, current_scalers = train_pipeline(X_train_base, Y_train, df_train_current, model_type=model_type)

    print(f"\n--- Evaluating {model_type} Model ---")
    model_results = {}
    model_predictions = {}
    for dataset_name, df_test in all_test_dfs.items():
        print(f"  on: {dataset_name} (Size: {len(df_test)})")
        accuracy, validation_rate, df_predictions = evaluate_pipeline(df_test, models, reducer, feature_cols, current_scalers)
        
        model_results[dataset_name] = {'Accuracy': accuracy, 'Validation_Rate': validation_rate}
        model_predictions[dataset_name] = df_predictions
        print(f"    Accuracy: {accuracy:.4f}, Validation Rate: {validation_rate:.4f}")
    
    all_artifacts[model_type] = {
        'models': models,
        'reducer': reducer,
        'scalers': current_scalers,
        'results': pd.DataFrame(model_results).T,
        'predictions': model_predictions
    }

# --- 5. Best Model Selection & Saving ---
print("\n\n--- Best Model Selection ---")
best_model_type = None
best_score = -1

for model_type, artifacts in all_artifacts.items():
    eval_results = artifacts['results'].loc['Evaluation_Set']
    score = eval_results['Accuracy'] * eval_results['Validation_Rate']
    print(f"Model: {model_type}, Score (Acc * Val): {score:.4f}")
    if score > best_score:
        best_score = score
        best_model_type = model_type

print(f"\nBest performing model is: {best_model_type} with a score of {best_score:.4f}")

# --- 6. Save Artifacts for the Best Model ---
print(f"\n--- Saving Artifacts for Best Model ({best_model_type}) ---")
best_artifacts = all_artifacts[best_model_type]
best_models = best_artifacts['models']
best_reducer = best_artifacts['reducer']
best_scalers = best_artifacts['scalers']
best_predictions = best_artifacts['predictions']
best_results_df = best_artifacts['results']

joblib.dump(best_models['s1'], os.path.join(OUTPUT_FOLDER, 'model_s1.joblib'))
best_models['s2'].save(os.path.join(OUTPUT_FOLDER, 'model_s2.keras'))
joblib.dump(best_reducer, os.path.join(OUTPUT_FOLDER, 'f_regression_reducer.joblib'))
joblib.dump(best_scalers, os.path.join(OUTPUT_FOLDER, 'scalers.joblib'))
print("Best model, reducer, and scalers saved.")

# Define columns for the comprehensive output file
base_info_cols = ['precursor_mz', 'formula', 'class', 'adduct', 'num_chain']
true_target_cols = ALL_TARGETS
pred_target_cols = [f'pred_{t}' for t in STAGE_2_TARGETS]
validation_col = ['is_valid_prediction']

for dataset_name, df_pred in best_predictions.items():
    if not df_pred.empty:
        # Ensure all columns exist in the dataframe before selecting
        cols_to_save = [col for col in base_info_cols + true_target_cols + pred_target_cols + validation_col if col in df_pred.columns]
        df_to_save = df_pred[cols_to_save]
        
        pred_filename = os.path.join(OUTPUT_FOLDER, f"predictions_{dataset_name}.csv")
        df_to_save.to_csv(pred_filename, index=False)
        print(f"Comprehensive predictions for {dataset_name} saved.")

# --- 7. Final Output ---
print("\n\n--- Final Results for Best Model ---")
print(f"\n--- Summary for {best_model_type} ---")
print(best_results_df.to_string(float_format="%.4f"))
results_filename = os.path.join(OUTPUT_FOLDER, "accuracy_summary_best_model.csv")
best_results_df.to_csv(results_filename)
print(f"\nAccuracy summary for the best model saved to: {results_filename}")


print("\n--- Script Finished ---")