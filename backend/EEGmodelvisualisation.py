#!/usr/bin/env python3
"""
Complete Model Visualization & Analysis for P300 EEGNet
Supports .h5 model files and .hdf5/.h5 dataset files
FIXED VERSION - Properly handles 'features' and 'erp_labels' keys
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import h5py  # For reading .hdf5 files
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style - with compatibility fix
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

class ModelVisualizer:
    """Comprehensive model visualization and analysis"""
    
    def __init__(self, model_path, data_path=None):
        """
        Initialize visualizer
        
        Args:
            model_path: Path to .h5 model file
            data_path: Path to test data file (.pkl, .hdf5, .h5) (optional)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.history = None
        
        print("="*60)
        print("P300 EEGNET MODEL ANALYSIS & VISUALIZATION")
        print("="*60)
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_test_data(self):
        """Load test data from pickle or HDF5 file"""
        if self.data_path is None:
            print("⚠️ No data path provided. Generating synthetic test data...")
            self.generate_synthetic_data()
            return True
        
        # Check file extension
        if self.data_path.endswith('.hdf5') or self.data_path.endswith('.h5'):
            return self.load_hdf5_data()
        elif self.data_path.endswith('.pkl') or self.data_path.endswith('.pickle'):
            return self.load_pickle_data()
        else:
            print(f"⚠️ Unknown file format: {self.data_path}")
            print("⚠️ Generating synthetic test data...")
            self.generate_synthetic_data()
            return True
    
    def load_hdf5_data(self):
        """Load test data from HDF5 file - FIXED VERSION"""
        try:
            print(f"Loading HDF5 file: {self.data_path}")
            with h5py.File(self.data_path, 'r') as f:
                # Print available keys
                all_keys = list(f.keys())
                print(f"Available keys in HDF5: {all_keys}")
                
                # Print details about each key to help identify X and y
                print("\nDataset details:")
                for key in all_keys:
                    print(f"  '{key}': shape={f[key].shape}, dtype={f[key].dtype}")
                
                # FIXED: For your specific dataset structure
                # X should be 'features' (the large 4D array)
                # y should be 'erp_labels' or 'target'
                
                x_key = 'features' if 'features' in f.keys() else None
                y_key = 'erp_labels' if 'erp_labels' in f.keys() else None
                
                # Fallback to target if erp_labels not found
                if y_key is None and 'target' in f.keys():
                    y_key = 'target'
                
                # If still not found, use the old search method
                if x_key is None:
                    possible_x_keys = ['X_test', 'x_test', 'test_data', 'test_x', 'epochs', 'data', 'X', 'features']
                    for key in possible_x_keys:
                        if key in f.keys():
                            x_key = key
                            break
                
                if y_key is None:
                    possible_y_keys = ['y_test', 'Y_test', 'test_labels', 'test_y', 'labels', 'y', 'Y', 'target', 'erp_labels']
                    for key in possible_y_keys:
                        if key in f.keys():
                            y_key = key
                            break
                
                if x_key is None or y_key is None:
                    print(f"\n❌ Could not find data keys automatically")
                    print(f"   x_key found: {x_key}")
                    print(f"   y_key found: {y_key}")
                    raise ValueError("Could not determine data keys")
                
                # Load the data
                print(f"\nLoading: X='{x_key}', y='{y_key}'")
                self.X_test = f[x_key][:]
                self.y_test = f[y_key][:]
                
                print(f"✅ Loaded from HDF5:")
                print(f"   Raw X shape: {self.X_test.shape}")
                print(f"   Raw y shape: {self.y_test.shape}")
            
            # Now reshape and process the data
            expected_shape = self.model.input_shape[1:]
            print(f"   Expected model input shape: {expected_shape}")
            
            # Handle different data formats for X
            if len(self.X_test.shape) == 2:
                # (samples, features) - need to reshape to (samples, channels, time, 1)
                n_samples = self.X_test.shape[0]
                n_channels = expected_shape[0]
                n_timepoints = expected_shape[1]
                
                if self.X_test.shape[1] == n_channels * n_timepoints:
                    self.X_test = self.X_test.reshape(n_samples, n_channels, n_timepoints, 1)
                    print(f"   Reshaped X from 2D to: {self.X_test.shape}")
                else:
                    print(f"   ⚠️  Warning: Feature dimension mismatch")
                    print(f"      Expected: {n_channels * n_timepoints}, Got: {self.X_test.shape[1]}")
            
            elif len(self.X_test.shape) == 3:
                # (samples, time, channels) or (samples, channels, time)
                n_channels = expected_shape[0]
                n_timepoints = expected_shape[1]
                
                if self.X_test.shape[1] == n_timepoints and self.X_test.shape[2] == n_channels:
                    print(f"   Transposing from (samples, time, channels) to (samples, channels, time)")
                    self.X_test = np.transpose(self.X_test, (0, 2, 1))
                
                self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
                print(f"   Final X shape after adding channel dim: {self.X_test.shape}")
            
            elif len(self.X_test.shape) == 4:
                n_channels = expected_shape[0]
                n_timepoints = expected_shape[1]
                
                if self.X_test.shape[1] == n_timepoints and self.X_test.shape[2] == n_channels:
                    print(f"   Transposing 4D from (samples, time, channels, 1) to (samples, channels, time, 1)")
                    self.X_test = np.transpose(self.X_test, (0, 2, 1, 3))
                print(f"   Final X shape: {self.X_test.shape}")
            
            # Ensure labels are 1D and binary
            if len(self.y_test.shape) > 1:
                self.y_test = self.y_test.flatten()
            
            # Convert labels to binary if needed
            unique_labels = np.unique(self.y_test)
            print(f"   Unique labels before conversion: {unique_labels}")
            
            if len(unique_labels) == 2 and 0 not in unique_labels:
                print(f"   Converting labels from {unique_labels} to [0, 1]")
                min_label = np.min(self.y_test)
                self.y_test = (self.y_test - min_label).astype(int)
            
            # Ensure same length
            if len(self.X_test) != len(self.y_test):
                print(f"   ⚠️  Length mismatch! X: {len(self.X_test)}, y: {len(self.y_test)}")
                min_len = min(len(self.X_test), len(self.y_test))
                self.X_test = self.X_test[:min_len]
                self.y_test = self.y_test[:min_len]
                print(f"   Truncated both to: {min_len}")
            
            # Use last 20% as test set
            if len(self.X_test) > 100:
                split_idx = int(len(self.X_test) * 0.8)
                print(f"   Using samples {split_idx} to {len(self.X_test)} as test set")
                self.X_test = self.X_test[split_idx:]
                self.y_test = self.y_test[split_idx:]
            
            print(f"\n✅ Final test set:")
            print(f"   X_test shape: {self.X_test.shape}")
            print(f"   y_test shape: {self.y_test.shape}")
            print(f"   Total samples: {len(self.y_test)}")
            
            if len(self.y_test) > 0:
                print(f"   Class distribution: {np.bincount(self.y_test.astype(int))}")
            else:
                raise ValueError("y_test has 0 samples!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading HDF5 data: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Generating synthetic test data...")
            self.generate_synthetic_data()
            return True
    
    def load_pickle_data(self):
        """Load test data from pickle file"""
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'X_test' in data and 'y_test' in data:
                self.X_test = data['X_test']
                self.y_test = data['y_test']
            else:
                # Try alternative formats
                self.X_test = data.get('epochs', None)
                self.y_test = data.get('labels', None)
                
                if self.X_test is not None:
                    # Split for testing
                    split_idx = int(len(self.X_test) * 0.8)
                    self.X_test = self.X_test[split_idx:]
                    self.y_test = self.y_test[split_idx:]
                    
                    # Reshape if needed
                    if len(self.X_test.shape) == 3:
                        self.X_test = self.X_test.reshape(*self.X_test.shape, 1)
            
            print(f"✅ Pickle data loaded: {self.X_test.shape}")
            print(f"   Labels: {self.y_test.shape}")
            print(f"   Class distribution: {np.bincount(self.y_test.astype(int))}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pickle data: {e}")
            print("⚠️ Generating synthetic test data...")
            self.generate_synthetic_data()
            return True
    
    def generate_synthetic_data(self):
        """Generate synthetic test data that mimics P300 characteristics"""
        print("Generating synthetic P300 test data...")
        
        # Model input shape from loaded model
        input_shape = self.model.input_shape[1:]  # (channels, samples, 1)
        n_samples = 500
        
        self.X_test = np.random.randn(n_samples, *input_shape)
        self.y_test = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% P300
        
        # Add P300-like features to target trials
        for i in range(n_samples):
            if self.y_test[i] == 1:
                # Add positive deflection around 300ms
                peak_sample = input_shape[1] // 2  # Middle of epoch
                self.X_test[i, :, peak_sample-5:peak_sample+5, 0] += np.random.randn(10) * 3
        
        print(f"✅ Generated synthetic data: {self.X_test.shape}")
        print(f"   P300 trials: {np.sum(self.y_test == 1)}")
        print(f"   Non-P300 trials: {np.sum(self.y_test == 0)}")
    
    def predict(self):
        """Generate predictions"""
        print("\nGenerating predictions...")
        
        self.y_pred_proba = self.model.predict(self.X_test, verbose=0)
        
        if len(self.y_pred_proba.shape) > 1 and self.y_pred_proba.shape[1] > 1:
            # Multi-class output
            self.y_pred = np.argmax(self.y_pred_proba, axis=1)
            self.y_pred_proba_positive = self.y_pred_proba[:, 1]  # P300 class
        else:
            # Binary output
            self.y_pred_proba_positive = self.y_pred_proba.flatten()
            self.y_pred = (self.y_pred_proba_positive > 0.5).astype(int)
        
        print(f"✅ Predictions generated")
        print(f"   Predicted P300: {np.sum(self.y_pred == 1)}")
        print(f"   Predicted Non-P300: {np.sum(self.y_pred == 0)}")
    
    def plot_model_architecture(self):
        """Plot model architecture summary"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Get model summary as text
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        # Display summary
        summary_text = '\n'.join(summary_lines[:30])  # First 30 lines
        ax.text(0.1, 0.9, summary_text, 
                fontfamily='monospace', fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.title('EEGNet Model Architecture', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('01_model_architecture.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 01_model_architecture.png")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    square=True, ax=ax, cbar_kws={'label': 'Count'},
                    xticklabels=['Non-P300', 'P300'],
                    yticklabels=['Non-P300', 'P300'])
        
        # Add percentages as text
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.7, f'({cm_normalized[i,j]:.1f}%)',
                       ha='center', va='center', fontsize=10, color='red')
        
        plt.title('Confusion Matrix\n(Count and Percentage)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('02_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 02_confusion_matrix.png")
        plt.close()
    
    def plot_classification_metrics(self):
        """Plot classification metrics bar chart"""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metrics.keys(), metrics.values(), 
                     color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Classification Performance Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)', linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('03_classification_metrics.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 03_classification_metrics.png")
        plt.close()
        
        return metrics
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba_positive)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
               label=f'Optimal threshold = {optimal_threshold:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('04_roc_curve.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 04_roc_curve.png")
        plt.close()
        
        return roc_auc, optimal_threshold
    
    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba_positive
        )
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='blue', lw=3,
               label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Baseline (proportion of positive class)
        baseline = np.sum(self.y_test == 1) / len(self.y_test)
        ax.axhline(y=baseline, color='red', linestyle='--', lw=2,
                  label=f'Baseline (P300 rate = {baseline:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig('05_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 05_precision_recall_curve.png")
        plt.close()
        
        return pr_auc
    
    def plot_prediction_distribution(self):
        """Plot distribution of prediction probabilities"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(self.y_pred_proba_positive[self.y_test == 0], 
                bins=50, alpha=0.7, label='Non-P300 (True)', color='blue', edgecolor='black')
        ax1.hist(self.y_pred_proba_positive[self.y_test == 1], 
                bins=50, alpha=0.7, label='P300 (True)', color='red', edgecolor='black')
        ax1.axvline(x=0.5, color='green', linestyle='--', lw=2, label='Decision Threshold')
        ax1.set_xlabel('Predicted P300 Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Predicted Probabilities', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        data_to_plot = [
            self.y_pred_proba_positive[self.y_test == 0],
            self.y_pred_proba_positive[self.y_test == 1]
        ]
        bp = ax2.boxplot(data_to_plot, labels=['Non-P300', 'P300'],
                        patch_artist=True, notch=True)
        
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Predicted P300 Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Probability Distribution by True Class', 
                     fontsize=14, fontweight='bold')
        ax2.axhline(y=0.5, color='green', linestyle='--', lw=2, label='Threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('06_prediction_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 06_prediction_distribution.png")
        plt.close()
    
    def plot_threshold_analysis(self):
        """Analyze performance at different thresholds"""
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_temp = (self.y_pred_proba_positive >= thresh).astype(int)
            accuracies.append(accuracy_score(self.y_test, y_pred_temp))
            
            # Handle edge cases
            if np.sum(y_pred_temp) > 0:
                precisions.append(precision_score(self.y_test, y_pred_temp, zero_division=0))
                recalls.append(recall_score(self.y_test, y_pred_temp, zero_division=0))
                f1_scores.append(f1_score(self.y_test, y_pred_temp, zero_division=0))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(thresholds, accuracies, label='Accuracy', lw=2, marker='o', markersize=3)
        ax.plot(thresholds, precisions, label='Precision', lw=2, marker='s', markersize=3)
        ax.plot(thresholds, recalls, label='Recall', lw=2, marker='^', markersize=3)
        ax.plot(thresholds, f1_scores, label='F1-Score', lw=2, marker='d', markersize=3)
        
        ax.axvline(x=0.5, color='red', linestyle='--', lw=2, label='Default Threshold (0.5)')
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresh = thresholds[optimal_idx]
        ax.axvline(x=optimal_thresh, color='green', linestyle='--', lw=2,
                  label=f'Optimal Threshold ({optimal_thresh:.3f})')
        
        ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics vs. Decision Threshold', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig('07_threshold_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 07_threshold_analysis.png")
        plt.close()
        
        return optimal_thresh
    
    def plot_class_wise_performance(self):
        """Detailed class-wise performance"""
        report = classification_report(self.y_test, self.y_pred, 
                                      target_names=['Non-P300', 'P300'],
                                      output_dict=True, zero_division=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Per-class metrics
        classes = ['Non-P300', 'P300']
        metrics = ['precision', 'recall', 'f1-score']
        
        data = np.array([[report[cls][metric] for metric in metrics] for cls in classes])
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i*width, data[:, i], width, 
                       label=metric.capitalize())
        
        axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Per-Class Performance Metrics', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(classes)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Support (sample count)
        support = [int(report[cls]['support']) for cls in classes]
        colors = ['#3498db', '#e74c3c']
        
        bars = axes[1].bar(classes, support, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        axes[1].set_title('Class Distribution in Test Set', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('08_class_wise_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 08_class_wise_performance.png")
        plt.close()
    
    def plot_error_analysis(self):
        """Analyze prediction errors"""
        errors = self.y_test != self.y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Error distribution
        error_counts = {
            'Correct': np.sum(~errors),
            'Errors': np.sum(errors)
        }
        
        axes[0, 0].pie(error_counts.values(), labels=error_counts.keys(),
                      autopct='%1.1f%%', startangle=90,
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Overall Prediction Accuracy', 
                            fontsize=14, fontweight='bold')
        
        # Error types
        false_positives = np.sum((self.y_test == 0) & (self.y_pred == 1))
        false_negatives = np.sum((self.y_test == 1) & (self.y_pred == 0))
        true_positives = np.sum((self.y_test == 1) & (self.y_pred == 1))
        true_negatives = np.sum((self.y_test == 0) & (self.y_pred == 0))
        
        error_types = ['True\nPositive', 'True\nNegative', 
                      'False\nPositive', 'False\nNegative']
        error_values = [true_positives, true_negatives, false_positives, false_negatives]
        error_colors = ['green', 'blue', 'orange', 'red']
        
        bars = axes[0, 1].bar(error_types, error_values, color=error_colors,
                             edgecolor='black', linewidth=1.5)
        axes[0, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Prediction Breakdown', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Probability distribution of errors
        correct_probs = self.y_pred_proba_positive[~errors]
        error_probs = self.y_pred_proba_positive[errors]
        
        axes[1, 0].hist(correct_probs, bins=30, alpha=0.7, 
                       label='Correct', color='green', edgecolor='black')
        axes[1, 0].hist(error_probs, bins=30, alpha=0.7, 
                       label='Errors', color='red', edgecolor='black')
        axes[1, 0].axvline(x=0.5, color='blue', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Probability Distribution: Correct vs Errors', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Confidence analysis
        confidence_levels = ['Very Low\n(0-0.2)', 'Low\n(0.2-0.4)', 
                            'Medium\n(0.4-0.6)', 'High\n(0.6-0.8)', 
                            'Very High\n(0.8-1.0)']
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        correct_dist = []
        error_dist = []
        
        for i in range(len(bins)-1):
            mask_correct = (correct_probs >= bins[i]) & (correct_probs < bins[i+1])
            mask_error = (error_probs >= bins[i]) & (error_probs < bins[i+1])
            correct_dist.append(np.sum(mask_correct))
            error_dist.append(np.sum(mask_error))
        
        x = np.arange(len(confidence_levels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, correct_dist, width, label='Correct',
                      color='green', edgecolor='black')
        axes[1, 1].bar(x + width/2, error_dist, width, label='Errors',
                      color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Confidence Analysis', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(confidence_levels, fontsize=9)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('09_error_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 09_error_analysis.png")
        plt.close()
    
    def plot_model_parameters(self):
        """Visualize model parameters and complexity"""
        layer_names = []
        param_counts = []
        layer_types = []
        
        for layer in self.model.layers:
            if layer.count_params() > 0:
                layer_names.append(layer.name)
                param_counts.append(layer.count_params())
                layer_types.append(layer.__class__.__name__)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Parameters per layer
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
        bars = axes[0].barh(layer_names, param_counts, color=colors, edgecolor='black')
        axes[0].set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
        axes[0].set_title('Parameters per Layer', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, param_counts)):
            axes[0].text(count, i, f' {count:,}', 
                        va='center', fontsize=9, fontweight='bold')
        
        # Model summary pie
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        param_summary = {
            f'Trainable\n({trainable_params:,})': trainable_params,
            f'Non-trainable\n({non_trainable_params:,})': non_trainable_params
        }
        
        axes[1].pie(param_summary.values(), labels=param_summary.keys(),
                   autopct='%1.1f%%', startangle=90,
                   colors=['lightblue', 'lightcoral'])
        axes[1].set_title(f'Model Parameters\nTotal: {total_params:,}', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('10_model_parameters.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 10_model_parameters.png")
        plt.close()
    
    def generate_summary_report(self, metrics, roc_auc, pr_auc, optimal_threshold):
        """Generate comprehensive text summary"""
        report_lines = [
            "="*80,
            "P300 EEGNET MODEL - PERFORMANCE SUMMARY REPORT",
            "="*80,
            "",
            f"Model Path: {self.model_path}",
            f"Test Samples: {len(self.y_test)}",
            f"P300 Samples: {np.sum(self.y_test == 1)} ({np.sum(self.y_test == 1)/len(self.y_test)*100:.1f}%)",
            f"Non-P300 Samples: {np.sum(self.y_test == 0)} ({np.sum(self.y_test == 0)/len(self.y_test)*100:.1f}%)",
            "",
            "-"*80,
            "CLASSIFICATION METRICS",
            "-"*80,
            f"Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)",
            f"Precision: {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)",
            f"Recall:    {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)",
            f"F1-Score:  {metrics['F1-Score']:.4f} ({metrics['F1-Score']*100:.2f}%)",
            "",
            "-"*80,
            "AREA UNDER CURVE (AUC) METRICS",
            "-"*80,
            f"ROC AUC:              {roc_auc:.4f}",
            f"Precision-Recall AUC: {pr_auc:.4f}",
            "",
            "-"*80,
            "THRESHOLD ANALYSIS",
            "-"*80,
            f"Default Threshold: 0.5000",
            f"Optimal Threshold: {optimal_threshold:.4f}",
            "",
            "-"*80,
            "CONFUSION MATRIX",
            "-"*80,
        ]
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        report_lines.extend([
            f"True Negatives:  {cm[0,0]:6d}  (Correctly predicted Non-P300)",
            f"False Positives: {cm[0,1]:6d}  (Non-P300 predicted as P300)",
            f"False Negatives: {cm[1,0]:6d}  (P300 predicted as Non-P300)",
            f"True Positives:  {cm[1,1]:6d}  (Correctly predicted P300)",
            "",
            "-"*80,
            "DETAILED CLASSIFICATION REPORT",
            "-"*80,
        ])
        
        # Add sklearn classification report
        report = classification_report(self.y_test, self.y_pred, 
                                       target_names=['Non-P300', 'P300'],
                                       zero_division=0)
        report_lines.append(report)
        
        report_lines.extend([
            "",
            "-"*80,
            "MODEL ARCHITECTURE",
            "-"*80,
            f"Total Parameters: {self.model.count_params():,}",
            f"Input Shape: {self.model.input_shape}",
            f"Output Shape: {self.model.output_shape}",
            "",
            "="*80,
            "END OF REPORT",
            "="*80
        ])
        
        # Save to file
        report_text = '\n'.join(report_lines)
        with open('00_model_summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Saved: 00_model_summary_report.txt")
        print("\n" + report_text)
    
    def create_comprehensive_dashboard(self):
        """Create a single comprehensive dashboard with all key metrics"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0:2])
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax1,
                   xticklabels=['Non-P300', 'P300'],
                   yticklabels=['Non-P300', 'P300'])
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Metrics Bar Chart
        ax2 = fig.add_subplot(gs[0, 2:4])
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        metrics = [accuracy, precision, recall, f1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        bars = ax2.bar(metric_names, metrics, color=colors, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax2.axhline(y=0.85, color='r', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[1, 0:2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba_positive)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', lw=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Precision-Recall Curve
        ax4 = fig.add_subplot(gs[1, 2:4])
        precision_curve, recall_curve, _ = precision_recall_curve(
            self.y_test, self.y_pred_proba_positive
        )
        pr_auc = auc(recall_curve, precision_curve)
        ax4.plot(recall_curve, precision_curve, lw=2, label=f'PR (AUC={pr_auc:.3f})')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Prediction Distribution
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.hist(self.y_pred_proba_positive[self.y_test == 0], 
                bins=40, alpha=0.6, label='Non-P300', color='blue')
        ax5.hist(self.y_pred_proba_positive[self.y_test == 1], 
                bins=40, alpha=0.6, label='P300', color='red')
        ax5.axvline(x=0.5, color='green', linestyle='--', lw=2)
        ax5.set_xlabel('Predicted P300 Probability')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Class Distribution
        ax6 = fig.add_subplot(gs[2, 2:4])
        class_counts = [np.sum(self.y_test == 0), np.sum(self.y_test == 1)]
        colors_pie = ['#3498db', '#e74c3c']
        ax6.pie(class_counts, labels=['Non-P300', 'P300'], autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax6.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        
        # Overall title
        fig.suptitle('P300 EEGNet Model - Comprehensive Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig('11_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 11_comprehensive_dashboard.png")
        plt.close()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("STARTING COMPLETE MODEL ANALYSIS")
        print("="*60 + "\n")
        
        # Load model
        if not self.load_model():
            return False
        
        # Load or generate test data
        if not self.load_test_data():
            return False
        
        # Generate predictions
        self.predict()
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        # Generate all plots
        self.plot_model_architecture()
        self.plot_confusion_matrix()
        metrics = self.plot_classification_metrics()
        roc_auc, optimal_threshold = self.plot_roc_curve()
        pr_auc = self.plot_precision_recall_curve()
        self.plot_prediction_distribution()
        optimal_thresh = self.plot_threshold_analysis()
        self.plot_class_wise_performance()
        self.plot_error_analysis()
        self.plot_model_parameters()
        self.create_comprehensive_dashboard()
        
        # Generate summary report
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60 + "\n")
        self.generate_summary_report(metrics, roc_auc, pr_auc, optimal_threshold)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  00_model_summary_report.txt")
        print("  01_model_architecture.png")
        print("  02_confusion_matrix.png")
        print("  03_classification_metrics.png")
        print("  04_roc_curve.png")
        print("  05_precision_recall_curve.png")
        print("  06_prediction_distribution.png")
        print("  07_threshold_analysis.png")
        print("  08_class_wise_performance.png")
        print("  09_error_analysis.png")
        print("  10_model_parameters.png")
        print("  11_comprehensive_dashboard.png")
        print("\n" + "="*60)
        
        return True


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='P300 EEGNet Model Visualization & Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With HDF5 dataset file
  python modelvi.py --model model.h5 --data dataset.hdf5
  
  # Without dataset (generates synthetic data)
  python modelvi.py --model model.h5
  
  # With pickle dataset
  python modelvi.py --model model.h5 --data test_data.pkl
        """
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to .h5 model file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to test data file (.hdf5, .h5, .pkl) (optional)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelVisualizer(args.model, args.data)
    
    # Run complete analysis
    success = visualizer.run_complete_analysis()
    
    if success:
        print("\n🎉 All visualizations generated successfully!")
        print("📊 Review the plots for your presentation!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    main()