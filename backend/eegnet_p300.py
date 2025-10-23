import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class EEGNet:
    """EEGNet implementation for P300 classification"""
    
    def __init__(self, nb_classes=2, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, 
                 dropoutType='Dropout'):
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutType = dropoutType
        
    def build_model(self):
        """Build EEGNet model"""
        input1 = keras.Input(shape=(self.Chans, self.Samples, 1))
        
        # Block 1
        block1 = layers.Conv2D(self.F1, (1, self.kernLength), padding='same',
                              input_shape=(self.Chans, self.Samples, 1),
                              use_bias=False)(input1)
        block1 = layers.BatchNormalization()(block1)
        block1 = layers.DepthwiseConv2D((self.Chans, 1), use_bias=False, 
                                       depth_multiplier=self.D,
                                       depthwise_constraint=keras.constraints.max_norm(1.))(block1)
        block1 = layers.BatchNormalization()(block1)
        block1 = layers.Activation('elu')(block1)
        block1 = layers.AveragePooling2D((1, 4))(block1)
        block1 = layers.Dropout(self.dropoutRate)(block1)
        
        # Block 2
        block2 = layers.SeparableConv2D(self.F2, (1, 16),
                                       use_bias=False, padding='same')(block1)
        block2 = layers.BatchNormalization()(block2)
        block2 = layers.Activation('elu')(block2)
        block2 = layers.AveragePooling2D((1, 8))(block2)
        block2 = layers.Dropout(self.dropoutRate)(block2)
        
        # Classification block
        flatten = layers.Flatten(name='flatten')(block2)
        dense = layers.Dense(self.nb_classes, name='dense', 
                           kernel_constraint=keras.constraints.max_norm(0.25))(flatten)
        softmax = layers.Activation('softmax', name='softmax')(dense)
        
        return keras.Model(inputs=input1, outputs=softmax)

class P300DataProcessor:
    """Data processor for P300 EEG data"""
    
    def __init__(self, sample_rate=256, epoch_length=0.6):
        self.sample_rate = sample_rate
        self.epoch_length = epoch_length
        self.epoch_samples = int(sample_rate * epoch_length)
        
    def load_bigp3_data(self, subject_path):
        """Load and preprocess BigP3 dataset"""
        # This would load the actual BigP3 data
        # For now, creating synthetic data structure
        data = {
            'eeg': np.random.randn(1000, 64, self.epoch_samples),  # trials x channels x samples
            'labels': np.random.randint(0, 2, 1000),  # P300 vs non-P300
            'characters': ['A', 'B', 'C'] * 333 + ['A'],  # Character being spelled
            'stimulus_codes': np.random.randint(1, 37, 1000)  # 6x6 matrix codes
        }
        return data
    
    def preprocess_eeg(self, raw_eeg):
        """Preprocess EEG data"""
        # Bandpass filter (0.1-30 Hz)
        processed = self.bandpass_filter(raw_eeg, 0.1, 30.0)
        
        # Baseline correction
        processed = self.baseline_correction(processed)
        
        # Standardization
        processed = self.standardize(processed)
        
        return processed
    
    def bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):  # For each trial
            for j in range(data.shape[1]):  # For each channel
                filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])
        
        return filtered_data
    
    def baseline_correction(self, data, baseline_window=(0, 0.1)):
        """Apply baseline correction"""
        baseline_samples = int(baseline_window[1] * self.sample_rate)
        baseline_mean = np.mean(data[:, :, :baseline_samples], axis=2, keepdims=True)
        return data - baseline_mean
    
    def standardize(self, data):
        """Standardize data across trials"""
        scaler = StandardScaler()
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_standardized = scaler.fit_transform(data_reshaped.T).T
        return data_standardized.reshape(original_shape)

class P300Trainer:
    """Training pipeline for P300 classification"""
    
    def __init__(self, model_config=None):
        self.model_config = model_config or {}
        self.model = None
        self.history = None
        
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the EEGNet model"""
        # Build model
        eegnet = EEGNet(**self.model_config)
        self.model = eegnet.build_model()
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint('best_p300_model.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))
        
        return test_acc, y_pred_classes

# Example usage
def main():
    # Initialize components
    processor = P300DataProcessor()
    trainer = P300Trainer({
        'nb_classes': 2,
        'Chans': 64,
        'Samples': 154,  # 0.6s at 256Hz
        'dropoutRate': 0.5
    })
    
    # Load and preprocess data
    print("Loading BigP3 dataset...")
    data = processor.load_bigp3_data("path/to/bigp3/subject1")
    
    print("Preprocessing EEG data...")
    X = processor.preprocess_eeg(data['eeg'])
    y = data['labels']
    
    # Reshape for EEGNet (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Train model
    print("Training EEGNet...")
    history = trainer.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate model
    print("Evaluating model...")
    test_acc, predictions = trainer.evaluate_model(X_test, y_test)
    
    print(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()