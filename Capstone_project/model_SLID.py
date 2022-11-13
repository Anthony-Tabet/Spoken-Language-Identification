import os
import librosa
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
import logging

class Spoken_Lang_Detection:
    def __init__(self, model_path):
        logging.info("Spoken Language detection class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")


    def features_extractor(file):
        audio, sample_rate = librosa.load(file)
        mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
        return mfccs_scaled_features

    def predict(self,path_to_audio):
        data  = Spoken_Lang_Detection.features_extractor(path_to_audio)
        data.reshape(1,-1)
        predictions = np.argmax(self.model.predict(np.array(data)),axis = 1)
        classes = ['Arabic','Chinese','Portuguese']
        return classes[predictions]