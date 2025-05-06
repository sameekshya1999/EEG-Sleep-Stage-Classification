🧠 EEG Sleep Stage Classification
This project focuses on classifying sleep stages using deep learning techniques applied to EEG signals. We explore different model architectures to compare performance on EEG data from sleep recordings.
📁 Project Structure
The project contains two primary experiments:

2channelcnn.py — Experiment 1: A CNN-based model trained on 2 EEG channels.
2eggnet_attention.py — Experiment 2: A modified EEGNet architecture with attention mechanism for improved sleep stage classification.

🚀 Getting Started
Follow these steps to set up and run the project locally.
1. Clone the Repository
git clone https://github.com/sameekshya1999/EEG-Sleep-Stage-Classification.git
cd EEG-Sleep-Stage-Classification

2. Create and Activate a Python Environment (Optional but Recommended)
python -m venv eeg-env
source eeg-env/bin/activate  # On Windows: eeg-env\Scripts\activate

3. Install the Required Packages
Make sure you have Python 3.8+ installed, then:
pip install -r requirements.txt

4. Run the Experiments
To run Experiment 1:
python 2channelcnn.py

To run Experiment 2:
python 2eggnet_attention.py

🧪 Requirements

TensorFlow
NumPy
Scikit-learn
Matplotlib
MNE
tqdm

These packages are listed in the requirements.txt file.

