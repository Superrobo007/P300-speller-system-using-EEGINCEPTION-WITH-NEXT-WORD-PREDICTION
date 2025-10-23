"""
BigP3 Dataset Processor for P300 Speller Training
Processes the BigP3 BCI dataset from PhysioNet for EEGNet training
Dataset: https://physionet.org/content/bigp3bci/1.0.0/
"""

import os
import numpy as np
import pandas as pd
import h5py
import scipy.io
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigP3Processor:
    """Process BigP3 dataset for P300 classification"""
    
    def __init__(self, dataset_path: str, output_path: str = "processed_data/"):
        """
        Initialize BigP3 processor
        
        Args:
            dataset_path: Path to BigP3 dataset directory
            output_path: Path to save processed data
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset parameters
        self.sample_rate = 256  # Hz
        self.epoch_start = -0.1  # seconds before stimulus
        self.epoch_end = 0.6     # seconds after stimulus
        self.epoch_length = self.epoch_end - self.epoch_start
        self.epoch_samples = int(self.epoch_length * self.sample_rate)
        
        # Channel information (typical EEG montage)
        self.eeg_channels = [
            'Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8'
        ]
        
        logger.info(f"Initialized BigP3 processor")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Epoch: {self.epoch_start}s to {self.epoch_end}s ({self.epoch_samples} samples)")
    
    def find_subject_files(self) -> List[Path]:
        """Find all subject data files in the dataset"""
        subject_files = []
        
        # Look for .mat files (common format for EEG datasets)
        mat_files = list(self.dataset_path.glob("**/*.mat"))
        subject_files.extend(mat_files)
        
        # Look for .edf files (EDF format)
        edf_files = list(self.dataset_path.glob("**/*.edf"))
        subject_files.extend(edf_files)
        
        # Look for .set files (EEGLAB format)
        set_files = list(self.dataset_path.glob("**/*.set"))
        subject_files.extend(set_files)
        
        logger.info(f"Found {len(subject_files)} data files")
        return sorted(subject_files)
    
    def load_subject_data(self, file_path: Path) -> Optional[Dict]:
        """
        Load data for a single subject
        
        Args:
            file_path: Path to subject data file
            
        Returns:
            Dictionary containing EEG data, events, and metadata
        """
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.mat':
                return self._load_mat_file(file_path)
            elif file_ext == '.edf':
                return self._load_edf_file(file_path)
            elif file_ext == '.set':
                return self._load_set_file(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _load_mat_file(self, file_path: Path) -> Dict:
        """Load MATLAB .mat file"""
        logger.info(f"Loading MAT file: {file_path}")
        
        data = scipy.io.loadmat(file_path)
        
        # Extract relevant fields (adapt based on actual BigP3 structure)
        if 'EEG' in data:
            eeg_data = data['EEG']
            if hasattr(eeg_data, 'data'):
                raw_data = eeg_data.data
                events = eeg_data.event if hasattr(eeg_data, 'event') else []
                sample_rate = float(eeg_data.srate) if hasattr(eeg_data, 'srate') else 256
            else:
                # Handle different MAT file structures
                raw_data = data.get('data', data.get('EEG', None))
                events = data.get('events', data.get('event', []))
                sample_rate = data.get('srate', data.get('fs', 256))
        else:
            # Try common field names
            raw_data = data.get('data', data.get('signal', None))
            events = data.get('events', data.get('triggers', []))
            sample_rate = data.get('fs', data.get('srate', 256))
        
        if raw_data is None:
            raise ValueError("Could not find EEG data in MAT file")
        
        return {
            'data': np.array(raw_data),
            'events': events,
            'sample_rate': float(sample_rate),
            'channels': self.eeg_channels[:raw_data.shape[0]],
            'file_path': str(file_path)
        }
    
    def _load_edf_file(self, file_path: Path) -> Dict:
        """Load EDF file using MNE"""
        logger.info(f"Loading EDF file: {file_path}")
        
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Get events
        events,_ = mne.find_events(raw, verbose=False)
        
        return {
            'data': raw.get_data(),
            'events': events,
            'sample_rate': raw.info['sfreq'],
            'channels': raw.ch_names,
            'file_path': str(file_path)
        }
    
    def _load_set_file(self, file_path: Path) -> Dict:
        """Load EEGLAB .set file using MNE"""
        logger.info(f"Loading SET file: {file_path}")
        
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        
        # Get events
        events= mne.find_events(raw, verbose=False)
        
        return {
            'data': raw.get_data(),
            'events': events,
            'sample_rate': raw.info['sfreq'],
            'channels': raw.ch_names,
            'file_path': str(file_path)
        }
    
    def extract_epochs(self, subject_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract P300 epochs from continuous EEG data
        
        Args:
            subject_data: Dictionary containing EEG data and events
            
        Returns:
            Tuple of (epochs, labels) where epochs is (n_epochs, n_channels, n_samples)
            and labels is (n_epochs,) with 1 for target, 0 for non-target
        """
        raw_data = subject_data['data']
        events = subject_data['events']
        sample_rate = subject_data['sample_rate']
        
        # Convert to numpy array if needed
        if not isinstance(events, np.ndarray):
            events = np.array(events)
        
        # Handle different event formats
        if events.ndim == 1:
            # Simple trigger array
            event_samples = np.where(events > 0)[0]
            event_codes = events[event_samples]
        elif events.ndim == 2:
            # MNE format: [sample, prev_val, event_code]
            if events.shape[1] >= 3:
                event_samples = events[:, 0]
                event_codes = events[:, 2]
            else:
                event_samples = events[:, 0]
                event_codes = events[:, 1]
        else:
            raise ValueError(f"Unsupported event format: {events.shape}")
        
        # Calculate epoch boundaries
        epoch_start_samples = int(self.epoch_start * sample_rate)
        epoch_end_samples = int(self.epoch_end * sample_rate)
        epoch_length_samples = epoch_end_samples - epoch_start_samples
        
        epochs = []
        labels = []
        
        for i, (sample, code) in enumerate(zip(event_samples, event_codes)):
            # Skip if epoch would be outside data bounds
            start_idx = int(sample + epoch_start_samples)
            end_idx = int(sample + epoch_end_samples)
            
            if start_idx < 0 or end_idx >= raw_data.shape[1]:
                continue
            
            # Extract epoch
            epoch = raw_data[:, start_idx:end_idx]
            
            # Determine if this is a target (P300) or non-target epoch
            # This depends on the specific event coding in BigP3
            # Common codes: 1=target, 2=non-target or similar
            is_target = self._is_target_event(code)
            
            epochs.append(epoch)
            labels.append(1 if is_target else 0)
        
        epochs = np.array(epochs)
        labels = np.array(labels)
        
        logger.info(f"Extracted {len(epochs)} epochs ({np.sum(labels)} targets, {len(labels) - np.sum(labels)} non-targets)")
        
        return epochs, labels
    
    def _is_target_event(self, event_code: int) -> bool:
        """
        Determine if event code represents a target stimulus
        
        This needs to be adapted based on the actual BigP3 event coding scheme
        """
        # Common P300 speller event codes:
        # - Target letters: specific codes (e.g., 1-36 for 6x6 matrix positions)
        # - Row/column flashes: different codes for target vs non-target
        
        # Placeholder logic - adapt based on actual BigP3 documentation
        target_codes = [1, 3, 5, 7, 9]  # Example target codes
        return int(event_code) in target_codes
    
    def preprocess_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """
        Preprocess epochs (filtering, baseline correction, normalization)
        
        Args:
            epochs: Raw epochs (n_epochs, n_channels, n_samples)
            
        Returns:
            Preprocessed epochs
        """
        logger.info("Preprocessing epochs...")
        
        # Apply bandpass filter (0.1-30 Hz)
        filtered_epochs = self._apply_bandpass_filter(epochs)
        
        # Baseline correction (use pre-stimulus period)
        baseline_corrected = self._baseline_correction(filtered_epochs)
        
        # Artifact rejection (simple amplitude-based)
        clean_epochs, valid_indices = self._reject_artifacts(baseline_corrected)
        
        # Standardization
        standardized_epochs = self._standardize_epochs(clean_epochs)
        
        logger.info(f"Preprocessing completed. {len(standardized_epochs)} epochs retained.")
        
        return standardized_epochs, valid_indices
    
    def _apply_bandpass_filter(self, epochs: np.ndarray, 
                              low_freq: float = 0.1, high_freq: float = 30.0) -> np.ndarray:
        """Apply bandpass filter to epochs"""
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        filtered_epochs = np.zeros_like(epochs)
        for i in range(epochs.shape[0]):
            for ch in range(epochs.shape[1]):
                filtered_epochs[i, ch, :] = filtfilt(b, a, epochs[i, ch, :])
        
        return filtered_epochs
    
    def _baseline_correction(self, epochs: np.ndarray, 
                           baseline_window: Tuple[float, float] = (-0.1, 0.0)) -> np.ndarray:
        """Apply baseline correction"""
        baseline_start = int((baseline_window[0] - self.epoch_start) * self.sample_rate)
        baseline_end = int((baseline_window[1] - self.epoch_start) * self.sample_rate)
        
        baseline_mean = np.mean(epochs[:, :, baseline_start:baseline_end], axis=2, keepdims=True)
        return epochs - baseline_mean
    
    def _reject_artifacts(self, epochs: np.ndarray, 
                         threshold: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Reject epochs with artifacts"""
        max_amplitudes = np.max(np.abs(epochs), axis=(1, 2))
        valid_indices = max_amplitudes < threshold
        
        clean_epochs = epochs[valid_indices]
        logger.info(f"Rejected {np.sum(~valid_indices)} epochs due to artifacts")
        
        return clean_epochs, valid_indices
    
    def _standardize_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """Standardize epochs across the dataset"""
        # Reshape for standardization
        n_epochs, n_channels, n_samples = epochs.shape
        epochs_reshaped = epochs.reshape(n_epochs, -1)
        
        # Standardize
        scaler = StandardScaler()
        epochs_standardized = scaler.fit_transform(epochs_reshaped)
        
        # Reshape back
        return epochs_standardized.reshape(n_epochs, n_channels, n_samples)
    
    def process_all_subjects(self) -> Dict:
        """Process all subjects in the dataset"""
        logger.info("Processing all subjects in BigP3 dataset...")
        
        subject_files = self.find_subject_files()
        all_epochs = []
        all_labels = []
        subject_info = []
        
        for i, file_path in enumerate(subject_files):
            logger.info(f"Processing subject {i+1}/{len(subject_files)}: {file_path.name}")
            
            # Load subject data
            subject_data = self.load_subject_data(file_path)
            if subject_data is None:
                continue
            
            # Extract epochs
            epochs, labels = self.extract_epochs(subject_data)
            if len(epochs) == 0:
                logger.warning(f"No epochs extracted for {file_path.name}")
                continue
            
            # Preprocess epochs
            clean_epochs, valid_indices = self.preprocess_epochs(epochs)
            clean_labels = labels[valid_indices]
            
            # Store data
            all_epochs.append(clean_epochs)
            all_labels.append(clean_labels)
            subject_info.append({
                'subject_id': i + 1,
                'file_path': str(file_path),
                'n_epochs': len(clean_epochs),
                'n_targets': np.sum(clean_labels),
                'n_non_targets': len(clean_labels) - np.sum(clean_labels)
            })
        
        # Combine all subjects
        if all_epochs:
            combined_epochs = np.concatenate(all_epochs, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            logger.info(f"Total processed epochs: {len(combined_epochs)}")
            logger.info(f"Target epochs: {np.sum(combined_labels)}")
            logger.info(f"Non-target epochs: {len(combined_labels) - np.sum(combined_labels)}")
            
            return {
                'epochs': combined_epochs,
                'labels': combined_labels,
                'subject_info': subject_info,
                'epoch_info': {
                    'sample_rate': self.sample_rate,
                    'epoch_start': self.epoch_start,
                    'epoch_end': self.epoch_end,
                    'epoch_samples': self.epoch_samples,
                    'channels': self.eeg_channels
                }
            }
        else:
            logger.error("No data could be processed!")
            return None
    
    def save_processed_data(self, processed_data: Dict, filename: str = "bigp3_processed.pkl"):
        """Save processed data to file"""
        output_file = self.output_path / filename
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Processed data saved to {output_file}")
        
        # Also save as separate numpy arrays for convenience
        np.save(self.output_path / "epochs.npy", processed_data['epochs'])
        np.save(self.output_path / "labels.npy", processed_data['labels'])
        
        # Save metadata as JSON
        import json
        metadata = {
            'subject_info': processed_data['subject_info'],
            'epoch_info': processed_data['epoch_info'],
            'data_shape': processed_data['epochs'].shape,
            'class_distribution': {
                'targets': int(np.sum(processed_data['labels'])),
                'non_targets': int(len(processed_data['labels']) - np.sum(processed_data['labels']))
            }
        }
        
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_train_test_split(self, processed_data: Dict, 
                               test_size: float = 0.2, val_size: float = 0.2):
        """Create train/validation/test splits"""
        epochs = processed_data['epochs']
        labels = processed_data['labels']
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            epochs, labels, test_size=test_size, random_state=42, 
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=42, stratify=y_train_val
        )
        
        # Add channel dimension for EEGNet (samples, channels, time, 1)
        X_train = X_train.reshape(*X_train.shape, 1)
        X_val = X_val.reshape(*X_val.shape, 1)
        X_test = X_test.reshape(*X_test.shape, 1)
        
        logger.info(f"Data splits created:")
        logger.info(f"Train: {X_train.shape[0]} samples")
        logger.info(f"Validation: {X_val.shape[0]} samples") 
        logger.info(f"Test: {X_test.shape[0]} samples")
        
        # Save splits
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        with open(self.output_path / "data_splits.pkl", 'wb') as f:
            pickle.dump(splits, f)
        
        return splits

def main():
    """Main function for BigP3 processing"""
    parser = argparse.ArgumentParser(description="Process BigP3 dataset for P300 classification")
    parser.add_argument("dataset_path", help="Path to BigP3 dataset directory")
    parser.add_argument("--output", default="processed_data/", 
                       help="Output directory for processed data")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for testing")  
    parser.add_argument("--val-size", type=float, default=0.2,
                       help="Proportion of data for validation")
    parser.add_argument("--subject-limit", type=int, default=None,
                       help="Limit number of subjects to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BigP3Processor(args.dataset_path, args.output)
    
    # Process all subjects
    processed_data = processor.process_all_subjects()
    
    if processed_data is None:
        logger.error("Failed to process dataset")
        return
    
    # Save processed data
    processor.save_processed_data(processed_data)
    
    # Create train/test splits
    splits = processor.create_train_test_split(
        processed_data, 
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Print summary
    print("\n" + "="*50)
    print("BigP3 Dataset Processing Complete!")
    print("="*50)
    print(f"Total epochs: {len(processed_data['epochs'])}")
    print(f"Target epochs: {np.sum(processed_data['labels'])}")
    print(f"Non-target epochs: {len(processed_data['labels']) - np.sum(processed_data['labels'])}")
    print(f"Data shape: {processed_data['epochs'].shape}")
    print(f"Sample rate: {processed_data['epoch_info']['sample_rate']} Hz")
    print(f"Epoch length: {processed_data['epoch_info']['epoch_samples']} samples")
    print(f"Channels: {len(processed_data['epoch_info']['channels'])}")
    print(f"\nData splits:")
    print(f"Train: {splits['X_train'].shape}")
    print(f"Validation: {splits['X_val'].shape}")
    print(f"Test: {splits['X_test'].shape}")
    print(f"\nFiles saved to: {processor.output_path}")

class BigP3Validator:
    """Validate processed BigP3 data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def validate_processed_data(self):
        """Validate the processed data"""
        print("Validating processed BigP3 data...")
        
        # Load processed data
        try:
            with open(self.data_path / "bigp3_processed.pkl", 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print("❌ Processed data file not found!")
            return False
        
        # Check data structure
        required_keys = ['epochs', 'labels', 'subject_info', 'epoch_info']
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing key: {key}")
                return False
        
        epochs = data['epochs']
        labels = data['labels']
        
        # Validate shapes
        if len(epochs.shape) != 3:
            print(f"❌ Invalid epochs shape: {epochs.shape} (expected 3D)")
            return False
        
        if epochs.shape[0] != len(labels):
            print(f"❌ Epoch-label mismatch: {epochs.shape[0]} vs {len(labels)}")
            return False
        
        # Check label distribution
        n_targets = np.sum(labels)
        n_non_targets = len(labels) - n_targets
        target_ratio = n_targets / len(labels)
        
        print(f"✅ Data validation passed!")
        print(f"   Epochs shape: {epochs.shape}")
        print(f"   Labels: {n_targets} targets, {n_non_targets} non-targets")
        print(f"   Target ratio: {target_ratio:.2%}")
        
        # Validate splits if they exist
        try:
            with open(self.data_path / "data_splits.pkl", 'rb') as f:
                splits = pickle.load(f)
            
            split_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            for key in split_keys:
                if key not in splits:
                    print(f"❌ Missing split key: {key}")
                    return False
            
            # Check if splits add up to original data
            total_samples = (len(splits['X_train']) + 
                           len(splits['X_val']) + 
                           len(splits['X_test']))
            
            if total_samples != len(epochs):
                print(f"❌ Split size mismatch: {total_samples} vs {len(epochs)}")
                return False
            
            print(f"✅ Data splits validation passed!")
            
        except FileNotFoundError:
            print("⚠️  Data splits file not found (not critical)")
            
        return True
    
    def plot_data_overview(self):
        """Create overview plots of the processed data"""
        try:
            import matplotlib.pyplot as plt
            
            # Load data
            with open(self.data_path / "bigp3_processed.pkl", 'rb') as f:
                data = pickle.load(f)
            
            epochs = data['epochs']
            labels = data['labels']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. Class distribution
            axes[0, 0].bar(['Non-target', 'Target'], 
                          [np.sum(labels == 0), np.sum(labels == 1)])
            axes[0, 0].set_title('Class Distribution')
            axes[0, 0].set_ylabel('Number of Epochs')
            
            # 2. Average target vs non-target ERP
            target_epochs = epochs[labels == 1]
            nontarget_epochs = epochs[labels == 0]
            
            # Average across channels and subjects
            target_avg = np.mean(target_epochs, axis=(0, 1))
            nontarget_avg = np.mean(nontarget_epochs, axis=(0, 1))
            
            time_axis = np.linspace(-0.1, 0.6, len(target_avg))
            axes[0, 1].plot(time_axis, target_avg, label='Target', linewidth=2)
            axes[0, 1].plot(time_axis, nontarget_avg, label='Non-target', linewidth=2)
            axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(0.3, color='r', linestyle='--', alpha=0.5, label='P300')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Amplitude (μV)')
            axes[0, 1].set_title('Average ERP')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Data quality metrics
            epoch_std = np.std(epochs, axis=2)  # Standard deviation across time
            axes[1, 0].hist(np.mean(epoch_std, axis=1), bins=50, alpha=0.7)
            axes[1, 0].set_xlabel('Average Epoch Std Dev')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Data Quality Distribution')
            
            # 4. Channel-wise signal power
            channel_power = np.mean(np.var(epochs, axis=2), axis=0)
            channels = data['epoch_info'].get('channels', [f'Ch{i+1}' for i in range(len(channel_power))])
            
            axes[1, 1].bar(range(len(channel_power)), channel_power)
            axes[1, 1].set_xlabel('Channel')
            axes[1, 1].set_ylabel('Signal Power')
            axes[1, 1].set_title('Channel-wise Signal Power')
            axes[1, 1].set_xticks(range(len(channels)))
            axes[1, 1].set_xticklabels(channels, rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.data_path / "data_overview.png", dpi=300, bbox_inches='tight')
            print(f"✅ Data overview plot saved to {self.data_path / 'data_overview.png'}")
            
        except ImportError:
            print("⚠️  Matplotlib not available - skipping plots")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")

def validate_dataset(data_path: str):
    """Validate processed dataset"""
    validator = BigP3Validator(data_path)
    
    if validator.validate_processed_data():
        validator.plot_data_overview()
        print("✅ Dataset validation completed successfully!")
        return True
    else:
        print("❌ Dataset validation failed!")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Validation mode
        data_path = sys.argv[2] if len(sys.argv) > 2 else "processed_data/"
        validate_dataset(data_path)
    else:
        # Processing mode
        main()