


'''
EEGNet with Attention for Sleep Stage Classification

This script implements a deep learning pipeline for classifying sleep stages
(Wake, N1, N2, N3, REM) using EEG data from the Sleep-EDF dataset. It includes:
- Memory-efficient data fetching and preprocessing with MNE
- A custom EEGNet model with a temporal attention layer
- On-the-fly and static augmentation for ~75,000-90,000 epochs
- Batch processing and mixed precision for GPU efficiency
- Comprehensive evaluation with accuracy, F1-score, and confusion matrix
- Visualizations of training curves

Optimized for 12GB RAM and 15GB GPU RAM, using ~5-7GB GPU memory.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import mne
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import gc

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level('ERROR')

NUM_SUBJECTS = 20
NUM_NIGHTS = 2
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
TARGET_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
EPOCH_DURATION = 30
BATCH_SIZE = 128
EPOCHS = 50
SAMPLING_RATE = 50

TELEMETRY_SUBJECTS = [2, 4, 5, 6, 7, 12, 13]

print(f"scikit-learn version: {sklearn.__version__}")

def fetch_data(subject_id, night, record_type='PSG'):
    try:
        dataset_id = subject_id + 1
        folder = "sleep-cassette" if night == 1 else "sleep-telemetry"

        if night == 1:
            prefix = f"SC4{dataset_id:02d}"
        else:
            if subject_id not in TELEMETRY_SUBJECTS:
                return None
            telemetry_map = {2: 702, 4: 704, 5: 705, 6: 706, 7: 707, 12: 712, 13: 713}
            prefix = f"ST{telemetry_map.get(subject_id, 700 + dataset_id)}"

        file_name = f"{prefix}{night if night == 1 else 2}E0-PSG.edf" if record_type == 'PSG' else \
                    f"{prefix}{night if night == 1 else 2}EC-Hypnogram.edf"
        url = f"{BASE_URL}{folder}/{file_name}"
        local_file = os.path.join("sleep_edf", file_name)
        os.makedirs("sleep_edf", exist_ok=True)

        if not os.path.exists(local_file):
            urllib.request.urlretrieve(url, local_file)
            print(f"Downloaded {file_name}")
        return local_file
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} fetching {file_name}: {e.reason}")
        return None
    except Exception as e:
        print(f"Error fetching {file_name}: {e}")
        return None

def get_available_subjects():
    available = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for subject_id in range(NUM_SUBJECTS):
            for night in range(1, NUM_NIGHTS + 1):
                futures.append((
                    subject_id,
                    night,
                    executor.submit(
                        lambda s, n: (
                            fetch_data(s, n, 'PSG') is not None and
                            fetch_data(s, n, 'Hypnogram') is not None
                        ),
                        subject_id, night
                    )
                ))

        for subject_id, night, future in tqdm(futures, desc="Checking availability"):
            if future.result():
                available.append((subject_id, night))
    print(f"Available subject-night pairs: {available}")
    return available

def augment_data(X):
    noise = np.random.normal(0, 0.01, X.shape)
    shift = np.random.randint(-50, 50)
    X_aug = np.roll(X + noise, shift, axis=1)
    return X_aug

def process_subject_night(subject_id, night):
    try:
        psg_file = fetch_data(subject_id, night, 'PSG')
        hypno_file = fetch_data(subject_id, night, 'Hypnogram')
        if psg_file is None or hypno_file is None:
            print(f"Skipping subject {subject_id}, night {night}: Missing files")
            return None, None

        raw = mne.io.read_raw_edf(psg_file, preload=False, verbose=False)
        available_channels = [ch for ch in TARGET_CHANNELS if ch in raw.ch_names]
        if not available_channels:
            print(f"No target channels for subject {subject_id}, night {night}")
            return None, None
        raw.pick_channels(available_channels)

        raw.load_data()
        raw.filter(0.5, 40.0, l_trans_bandwidth=0.5, h_trans_bandwidth=10.0, verbose=False)
        raw.resample(SAMPLING_RATE, npad="auto")

        events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_DURATION)
        epochs_mne = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_DURATION-1/raw.info['sfreq'],
                                picks=available_channels, baseline=None, preload=True)
        data = epochs_mne.get_data(units='uV')

        annotations = mne.read_annotations(hypno_file)
        labels = np.zeros(len(epochs_mne), dtype=int)
        stage_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4
        }

        for annot in annotations:
            onset = int(annot['onset'] / EPOCH_DURATION)
            duration = int(annot['duration'] / EPOCH_DURATION)
            stage = annot['description']
            if stage in stage_map:
                for i in range(max(0, onset), min(len(epochs_mne), onset + duration)):
                    labels[i] = stage_map[stage]

        data = (
            (data - np.mean(data, axis=(1, 2), keepdims=True)) /
            np.std(data, axis=(1, 2), keepdims=True)
        )
        X = data.transpose(0, 2, 1)
        X_aug = augment_data(X)
        X = np.concatenate([X, X_aug])
        labels = np.concatenate([labels, labels])

        del raw, epochs_mne, data
        gc.collect()

        print(f"Processed subject {subject_id}, night {night}: {X.shape[0]} epochs")
        return X, labels
    except Exception as e:
        print(f"Error processing subject {subject_id}, night {night}: {e}")
        return None, None

class EEGDataGenerator(Sequence):
    def __init__(self, X, y, batch_size, augment=True, class_weights=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.batch_size = batch_size
        self.augment = augment
        self.class_weights = class_weights

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]

        if self.augment:
            X_batch = augment_data(X_batch).astype(np.float32)

        sample_weights = np.ones_like(y_batch, dtype=np.float32)
        if self.class_weights:
            sample_weights = np.array([self.class_weights[label] for label in y_batch], dtype=np.float32)

        return X_batch, y_batch, sample_weights

class TemporalAttention(layers.Layer):
    def __init__(self, heads=4, key_dim=24):
        super().__init__()
        self.multi_head = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)
        self.norm = layers.LayerNormalization()
        self.add = layers.Add()

    def call(self, inputs):
        attn_output = self.multi_head(inputs, inputs)
        out = self.add([inputs, attn_output])
        return self.norm(out)

def build_eegnet_attention(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, 7, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = TemporalAttention(heads=4, key_dim=24)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(5, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average=None)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("\nPer-class Metrics:")
    for i, stage in enumerate(stage_names):
        print(f"{stage}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stage_names, yticklabels=stage_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def data_generator(available, batch_size=2000):
    for subject_id, night in available:
        X, y = process_subject_night(subject_id, night)
        if X is None or y is None:
            continue
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
        del X, y
        gc.collect()

def run_pipeline():
    available = get_available_subjects()
    if not available:
        return

    X_train, y_train, X_test, y_test = [], [], [], []
    for X_batch, y_batch in tqdm(data_generator(available), desc="Processing data"):
        if X_batch is None or y_batch is None:
            continue
        class_counts = np.bincount(y_batch)
        stratify = y_batch if min(class_counts[class_counts > 0]) >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_batch, y_batch, test_size=0.2, stratify=stratify, random_state=42)
        X_train.append(X_tr); y_train.append(y_tr)
        X_test.append(X_te); y_test.append(y_te)
        del X_batch, y_batch
        gc.collect()

    if not X_train:
        return

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_eegnet_attention(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_generator = EEGDataGenerator(X_train, y_train, BATCH_SIZE, augment=True, class_weights=class_weight_dict)
    val_generator = EEGDataGenerator(X_test, y_test, BATCH_SIZE, augment=False, class_weights=class_weight_dict)

    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, verbose=1)

    plot_training_curves(history)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()





