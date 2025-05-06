import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mne
import urllib.request
import os
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Suppress MNE warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level('ERROR')

# Constants
NUM_SUBJECTS = 20
NUM_NIGHTS = 2
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"

def fetch_data(subject_id, night, record_type='PSG'):
    try:
        if record_type == 'PSG':
            file_name = f"SC4{subject_id:02d}{night}E0-PSG.edf"
        else:
            file_name = f"SC4{subject_id:02d}{night}EC-Hypnogram.edf"

        url = BASE_URL + ("sleep-cassette/" if night == 1 else "sleep-telemetry/") + file_name
        local_file = f"sleep_edf/{file_name}"
        os.makedirs("sleep_edf", exist_ok=True)

        if not os.path.exists(local_file):
            urllib.request.urlretrieve(url, local_file)
            print(f"Downloaded: {file_name}")

        return local_file
    except Exception:
        return None

def get_available_subjects():
    available = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for subject_id in range(NUM_SUBJECTS):
            for night in range(1, NUM_NIGHTS+1):
                futures.append((subject_id, night, executor.submit(
                    lambda s, n: fetch_data(s, n) is not None, subject_id, night)))
        for subject_id, night, future in tqdm(futures, desc="Checking availability"):
            if future.result():
                available.append((subject_id, night))
    return available

def process_subject_night(subject_id, night):
    try:
        psg_file = fetch_data(subject_id, night, 'PSG')
        hypno_file = fetch_data(subject_id, night, 'Hypnogram')
        if psg_file is None or hypno_file is None:
            return None, None

        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        required_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
        available = [ch for ch in required_channels if ch in raw.ch_names]
        if len(available) < 2:
            print(f"Skipping subject {subject_id}, night {night}: missing required channels")
            return None, None
        raw.pick_channels(available)

        raw.filter(0.5, 40.0, l_trans_bandwidth=0.5, h_trans_bandwidth=10.0, verbose=False)
        data = raw.get_data(units='uV')
        sfreq = raw.info['sfreq']

        samples_per_epoch = int(30 * sfreq)
        n_epochs = data.shape[1] // samples_per_epoch
        epochs = np.array([data[:, i*samples_per_epoch:(i+1)*samples_per_epoch] for i in range(n_epochs)])

        annotations = mne.read_annotations(hypno_file)
        labels = np.zeros(n_epochs, dtype=int)
        stage_map = {
            'Sleep stage W': 0, 'Sleep stage 1': 1,
            'Sleep stage 2': 2, 'Sleep stage 3': 3,
            'Sleep stage 4': 3, 'Sleep stage R': 4
        }

        for annot in annotations:
            onset = int(annot['onset'] / 30)
            duration = int(annot['duration'] / 30)
            stage = annot['description']
            if stage in stage_map:
                for i in range(max(0, onset), min(n_epochs, onset + duration)):
                    labels[i] = stage_map[stage]

        epochs = (epochs - np.mean(epochs, axis=(1,2), keepdims=True)) / np.std(epochs, axis=(1,2), keepdims=True)
        X = epochs.transpose(0, 2, 1)
        y = labels

        return X, y
    except Exception as e:
        print(f"Error processing subject {subject_id} night {night}: {str(e)}")
        return None, None

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    available = get_available_subjects()
    print(f"\nFound {len(available)} available subject-night combinations")
    if not available:
        print("No data available - check your internet connection")
        return

    all_X, all_y = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_subject_night, s, n) for s, n in available]
        for future in tqdm(futures, desc="Processing data"):
            X, y = future.result()
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)

    if not all_X:
        print("No valid data processed")
        return

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    print(f"\nFinal dataset: {X.shape[0]} epochs")
    print("Class distribution:")
    for i, stage in enumerate(['Wake', 'N1', 'N2', 'N3', 'REM']):
        print(f"{stage}: {np.sum(y == i)} ({(np.sum(y == i)/len(y))*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model((X.shape[1], X.shape[2]))
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Per-class metrics
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('2channel_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
