import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# -------------------------
# CONFIGURATION - FIXED FOR BETTER PERFORMANCE
# -------------------------
CONFIG = {
    'dataset_path': r"E:\bio_proj\Code\Project\GIB-UVA ERP-BCI.hdf5",
    'output_dir': r"E:\bio_proj\Code\Project\newmod",
    
    # Training params - OPTIMIZED
    'epochs': 50,  # Reduced from 100 (faster, early stopping will handle it)
    'batch_size': 32,  # Changed from 64 - better for stability
    'learning_rate': 0.001,
    'test_split': 0.2,
    
    # Data selection - CRITICAL FIX!
    'use_subjects': None,  # Use ALL subjects
    'max_sequences': None,  # FIXED: Use ALL sequences (was limiting data)
    
    # Model params - OPTIMIZED
    'dropoutRate': 0.3,  # Reduced from 0.5 (less aggressive regularization)
    
    # Class imbalance handling - CRITICAL FIX!
    'balance_strategy': 'undersample',  # Changed from 'weighted' - more stable
    'undersample_ratio': 1.5,  # 1.5:1 ratio (slightly imbalanced, more realistic)
    
    # Data augmentation - NEW!
    'use_augmentation': True,
    'augmentation_factor': 1.5,  # Increase minority class by 50%
}

# -------------------------
# DATA LOADING
# -------------------------
def load_dataset(hdf5_path, use_subjects=None, max_sequences=None):
    """Load ERP-based speller dataset from HDF5"""
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Load all variables
        features = f['features'][:]  # [n_stimuli, n_samples, n_channels]
        erp_labels = f['erp_labels'][:].flatten()  # [n_stimuli] - CORRECT LABEL!
        subjects = f['subjects'][:].flatten()  # [n_stimuli]
        sequences = f['sequences'][:].flatten()  # [n_stimuli]
        
        print(f"\nDataset shape: {features.shape}")
        print(f"  Total samples: {features.shape[0]:,}")
        print(f"  Time points: {features.shape[1]}")
        print(f"  Channels: {features.shape[2]}")
        print(f"  Channel order: ['FZ', 'CZ', 'PZ', 'P3', 'P4', 'PO7', 'PO8', 'OZ']")
        
        # Class distribution
        n_target = np.sum(erp_labels == 1)
        n_nontarget = np.sum(erp_labels == 0)
        imbalance_ratio = n_nontarget / n_target if n_target > 0 else 0
        
        print(f"\nClass distribution (erp_labels):")
        print(f"  Target (P300):     {n_target:,} ({n_target/len(erp_labels)*100:.2f}%)")
        print(f"  Non-target:        {n_nontarget:,} ({n_nontarget/len(erp_labels)*100:.2f}%)")
        print(f"  Imbalance ratio:   {imbalance_ratio:.2f}:1")
        
        # Subject info
        unique_subjects = np.unique(subjects)
        print(f"\nSubjects: {len(unique_subjects)} total")
        
    # Filter by subjects if specified
    if use_subjects is not None:
        mask = np.isin(subjects, use_subjects)
        features = features[mask]
        erp_labels = erp_labels[mask]
        subjects = subjects[mask]
        sequences = sequences[mask]
        print(f"\nFiltered to {len(use_subjects)} subjects: {len(features):,} samples")
    
    # Filter by sequences if specified
    if max_sequences is not None:
        mask = sequences <= max_sequences
        features = features[mask]
        erp_labels = erp_labels[mask]
        subjects = subjects[mask]
        sequences = sequences[mask]
        print(f"Filtered to first {max_sequences} sequences: {len(features):,} samples")
    else:
        print(f"\n✅ Using ALL sequences: {len(features):,} samples")
    
    return features, erp_labels, subjects, sequences

# -------------------------
# DATA PREPROCESSING
# -------------------------
def preprocess_data(features, erp_labels, subjects, test_split=0.2):
    """Split data by subjects and prepare for training"""
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    n_test_subjects = max(1, int(len(unique_subjects) * test_split))
    
    # Random split subjects
    np.random.seed(42)
    test_subjects = np.random.choice(unique_subjects, size=n_test_subjects, replace=False)
    train_subjects = np.setdiff1d(unique_subjects, test_subjects)
    
    print(f"\nSubject split:")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Test subjects:  {len(test_subjects)}")
    
    # Split data
    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    X_train = features[train_mask]
    y_train = erp_labels[train_mask]
    X_test = features[test_mask]
    y_test = erp_labels[test_mask]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples ({np.sum(y_train==1):,} P300, {np.sum(y_train==0):,} Non-P300)")
    print(f"  Test:  {len(X_test):,} samples ({np.sum(y_test==1):,} P300, {np.sum(y_test==0):,} Non-P300)")
    
    # Normalize data - IMPORTANT!
    print("\nNormalizing data (z-score)...")
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)
    X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
    print(f"  Train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"  Test:  mean={X_test.mean():.4f}, std={X_test.std():.4f}")
    
    # Reshape for EEGNet: (samples, channels, time_points, 1)
    X_train = np.transpose(X_train, (0, 2, 1))  # (samples, channels, time_points)
    X_test = np.transpose(X_test, (0, 2, 1))
    X_train = np.expand_dims(X_train, axis=-1)  # Add kernel dimension
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"\nReshaped for EEGNet:")
    print(f"  X_train: {X_train.shape} (samples, channels, time_points, 1)")
    print(f"  X_test:  {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

# -------------------------
# DATA AUGMENTATION - NEW!
# -------------------------
def augment_minority_class(X_train, y_train, factor=1.5):
    """Augment minority class with noise and slight variations"""
    print("\n" + "="*70)
    print("DATA AUGMENTATION")
    print("="*70)
    
    # Find minority class samples
    minority_idx = np.where(y_train == 1)[0]
    n_minority = len(minority_idx)
    n_augment = int(n_minority * (factor - 1))
    
    if n_augment <= 0:
        print("No augmentation needed")
        return X_train, y_train
    
    print(f"\nAugmenting minority class (P300):")
    print(f"  Original P300 samples: {n_minority}")
    print(f"  Augmented samples to add: {n_augment}")
    
    # Randomly select samples to augment
    np.random.seed(42)
    augment_idx = np.random.choice(minority_idx, size=n_augment, replace=True)
    
    # Create augmented samples
    X_augmented = X_train[augment_idx].copy()
    
    # Add small random noise
    noise = np.random.normal(0, 0.05, X_augmented.shape)
    X_augmented += noise
    
    # Random time shift (±2 samples)
    for i in range(len(X_augmented)):
        shift = np.random.randint(-2, 3)
        if shift != 0:
            X_augmented[i] = np.roll(X_augmented[i], shift, axis=1)
    
    # Random amplitude scaling (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1, (len(X_augmented), 1, 1, 1))
    X_augmented *= scale
    
    # Combine with original data
    X_train = np.concatenate([X_train, X_augmented], axis=0)
    y_train = np.concatenate([y_train, np.ones(n_augment, dtype=y_train.dtype)], axis=0)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    print(f"\nAfter augmentation:")
    print(f"  Total: {len(y_train):,} samples")
    print(f"  P300: {np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"  Non-P300: {np.sum(y_train==0):,} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    
    return X_train, y_train

# -------------------------
# CLASS IMBALANCE HANDLING - IMPROVED
# -------------------------
def handle_imbalance(X_train, y_train, strategy='undersample', ratio=1.0):
    """Handle class imbalance using different strategies"""
    print("\n" + "="*70)
    print(f"HANDLING IMBALANCE - Strategy: {strategy.upper()}")
    print("="*70)
    
    print(f"\nBefore balancing:")
    print(f"  Total: {len(y_train):,} samples")
    print(f"  P300: {np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"  Non-P300: {np.sum(y_train==0):,} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
    
    if strategy == 'weighted':
        # Use class weights (no resampling)
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(y_train), 
                                             y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"\nClass weights computed:")
        print(f"  Non-target (0): {class_weight_dict[0]:.3f}")
        print(f"  Target (1):     {class_weight_dict[1]:.3f}")
        return X_train, y_train, class_weight_dict
    
    elif strategy == 'undersample':
        # Undersample majority class
        target_idx = np.where(y_train == 1)[0]
        nontarget_idx = np.where(y_train == 0)[0]
        
        n_target = len(target_idx)
        n_nontarget_desired = int(n_target * ratio)
        
        # Ensure we don't try to select more samples than available
        n_nontarget_desired = min(n_nontarget_desired, len(nontarget_idx))
        
        # Randomly select non-target samples
        np.random.seed(42)
        selected_nontarget = np.random.choice(nontarget_idx, 
                                             size=n_nontarget_desired, 
                                             replace=False)
        
        # Combine and shuffle
        selected_idx = np.concatenate([target_idx, selected_nontarget])
        np.random.shuffle(selected_idx)
        
        X_train = X_train[selected_idx]
        y_train = y_train[selected_idx]
        
        print(f"\nAfter undersampling:")
        print(f"  Total: {len(y_train):,} samples")
        print(f"  P300: {np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
        print(f"  Non-P300: {np.sum(y_train==0):,} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
        print(f"  Ratio: {np.sum(y_train==0)/np.sum(y_train==1):.2f}:1")
        
        return X_train, y_train, None
    
    elif strategy == 'oversample':
        # Oversample minority class
        target_idx = np.where(y_train == 1)[0]
        nontarget_idx = np.where(y_train == 0)[0]
        
        n_nontarget = len(nontarget_idx)
        n_target_desired = int(n_nontarget / ratio)
        
        # Oversample with replacement
        np.random.seed(42)
        oversampled_target = np.random.choice(target_idx, 
                                             size=n_target_desired, 
                                             replace=True)
        
        # Combine and shuffle
        all_idx = np.concatenate([nontarget_idx, oversampled_target])
        np.random.shuffle(all_idx)
        
        X_train = X_train[all_idx]
        y_train = y_train[all_idx]
        
        print(f"\nAfter oversampling:")
        print(f"  Total: {len(y_train):,} samples")
        print(f"  P300: {np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
        print(f"  Non-P300: {np.sum(y_train==0):,} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)")
        
        return X_train, y_train, None
    
    else:  # 'none'
        print("\nNo balancing applied")
        return X_train, y_train, None

# -------------------------
# EEGNET MODEL - IMPROVED
# -------------------------
def build_eegnet(n_channels=8, n_samples=128, dropoutRate=0.3):
    """
    Build EEGNet model adapted for ERP/P300 detection
    IMPROVED: Better regularization and architecture
    """
    input_layer = keras.Input(shape=(n_channels, n_samples, 1))
    
    # Block 1: Temporal convolution
    block1 = layers.Conv2D(8, (1, 64), padding='same', 
                          use_bias=False,
                          kernel_regularizer=keras.regularizers.l2(0.0001))(input_layer)
    block1 = layers.BatchNormalization()(block1)
    
    # Depthwise convolution (spatial filtering)
    block1 = layers.DepthwiseConv2D((n_channels, 1), 
                                   use_bias=False, 
                                   depth_multiplier=2,
                                   depthwise_constraint=keras.constraints.max_norm(1.),
                                   depthwise_regularizer=keras.regularizers.l2(0.0001))(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = layers.Dropout(dropoutRate)(block1)
    
    # Block 2: Separable convolution
    block2 = layers.SeparableConv2D(16, (1, 16), 
                                   use_bias=False, 
                                   padding='same',
                                   depthwise_regularizer=keras.regularizers.l2(0.0001),
                                   pointwise_regularizer=keras.regularizers.l2(0.0001))(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1, 8))(block2)
    block2 = layers.Dropout(dropoutRate)(block2)
    
    # Classification layer
    flatten = layers.Flatten()(block2)
    dense = layers.Dense(2, 
                        kernel_constraint=keras.constraints.max_norm(0.25),
                        kernel_regularizer=keras.regularizers.l2(0.0001))(flatten)
    output = layers.Activation('softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# -------------------------
# TRAINING - IMPROVED
# -------------------------
def train_model():
    """Main training function with improvements"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    model_path = os.path.join(CONFIG['output_dir'], f"p300_eegnet_{timestamp}.h5")
    
    print("\n" + "="*70)
    print("P300 SPELLER TRAINING - EEGNet (IMPROVED)")
    print("="*70)
    print(f"Timestamp: {timestamp}")
    
    # 1. Load dataset
    features, erp_labels, subjects, sequences = load_dataset(
        CONFIG['dataset_path'],
        use_subjects=CONFIG['use_subjects'],
        max_sequences=CONFIG['max_sequences']
    )
    
    # 2. Preprocess and split
    X_train, y_train, X_test, y_test = preprocess_data(
        features, erp_labels, subjects, CONFIG['test_split']
    )
    
    # 3. Data augmentation (if enabled)
    if CONFIG['use_augmentation']:
        X_train, y_train = augment_minority_class(
            X_train, y_train, 
            factor=CONFIG['augmentation_factor']
        )
    
    # 4. Handle class imbalance
    X_train, y_train, class_weight_dict = handle_imbalance(
        X_train, y_train, 
        strategy=CONFIG['balance_strategy'],
        ratio=CONFIG['undersample_ratio']
    )
    
    # 5. Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    
    model = build_eegnet(
        n_channels=n_channels,
        n_samples=n_samples,
        dropoutRate=CONFIG['dropoutRate']
    )
    
    # Use Adam with gradient clipping for stability
    optimizer = keras.optimizers.Adam(
        learning_rate=CONFIG['learning_rate'],
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel input shape: (batch, {n_channels}, {n_samples}, 1)")
    print(f"Total parameters: {model.count_params():,}")
    print("\nModel architecture:")
    model.summary()
    
    # 6. Setup callbacks - IMPROVED
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, 
            monitor='val_accuracy',  # Changed from val_loss
            save_best_only=True, 
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Changed from val_loss
            patience=20,  # Increased patience
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',  # Changed from val_loss
            factor=0.5, 
            patience=7, 
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
    ]
    
    # 7. Train
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Dropout rate: {CONFIG['dropoutRate']}")
    print(f"Training samples: {len(X_train):,}")
    print("="*70 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,  # Reduced from 0.2 for more training data
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=['Non-Target', 'Target (P300)'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Non-Target  Target")
    print(f"Actual Non-Target    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"       Target        {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    # Detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        auc_roc = 0.0
    
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:     {auc_roc:.4f}")
    
    # Performance interpretation
    print("\n" + "="*70)
    print("PERFORMANCE ASSESSMENT")
    print("="*70)
    
    if accuracy > 0.85 and recall > 0.75 and precision > 0.60:
        print("✅ EXCELLENT - Model is ready for deployment!")
        print("  High accuracy and balanced precision/recall")
    elif accuracy > 0.75 and recall > 0.65 and f1 > 0.60:
        print("✅ GOOD - Model should work well in practice")
        print("  Acceptable performance for P300 detection")
    elif accuracy > 0.70 and f1 > 0.50:
        print("⚠️  ACCEPTABLE - Model is functional")
        print("  May benefit from hyperparameter tuning")
    else:
        print("❌ NEEDS IMPROVEMENT")
        print("  Suggestions:")
        print("  - Verify data loading (check erp_labels)")
        print("  - Try different balance_strategy")
        print("  - Adjust augmentation_factor")
        print("  - Check for data quality issues")
    
    print("="*70)
    
    # 9. Save results
    print(f"\n✅ Model saved: {model_path}")
    
    # Save training history
    history_path = model_path.replace('.h5', '_history.json')
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✅ History saved: {history_path}")
    
    # Save evaluation metrics
    metrics_path = model_path.replace('.h5', '_metrics.json')
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': cm.tolist(),
        'config': CONFIG
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    # 10. Plot training curves
    plot_training_history(history, model_path)
    
    return model, history, metrics

# -------------------------
# PLOTTING - IMPROVED
# -------------------------
def plot_training_history(history, model_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[0, 1]
    ax.plot(history.history['loss'], label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    if 'lr' in history.history:
        ax.plot(history.history['lr'], linewidth=2, color='orange')
        ax.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
    
    # Final metrics summary
    ax = axes[1, 1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    
    summary_text = (
        f'Training Summary\n\n'
        f'Final Train Acc: {final_train_acc:.4f}\n'
        f'Final Val Acc: {final_val_acc:.4f}\n\n'
        f'Best Val Acc: {best_val_acc:.4f}\n'
        f'Best Epoch: {best_epoch}\n\n'
        f'Total Epochs: {len(history.history["accuracy"])}'
    )
    
    ax.text(0.5, 0.5, summary_text,
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plot_path = model_path.replace('.h5', '_training.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plot saved: {plot_path}")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("P300 SPELLER TRAINING WITH EEGNET - IMPROVED VERSION")
    print("="*70)
    print("\n🔧 KEY IMPROVEMENTS:")
    print("  ✅ Uses ALL sequences (no artificial data limitation)")
    print("  ✅ Undersampling for better class balance")
    print("  ✅ Data augmentation for minority class")
    print("  ✅ Data normalization (z-score)")
    print("  ✅ Reduced dropout (0.3 instead of 0.5)")
    print("  ✅ L2 regularization added")
    print("  ✅ Gradient clipping for stability")
    print("  ✅ Monitors val_accuracy (not val_loss)")
    print("\n" + "="*70)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        if key == 'dataset_path' or key == 'output_dir':
            print(f"  {key}:")
            print(f"    {value}")
        else:
            print(f"  {key}: {value}")
    print("="*70)
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("  1. This script uses erp_labels (NOT target)")
    print("  2. All sequences will be used (max_sequences = None)")
    print("  3. Expected accuracy: 80-90% (not 69%)")
    print("  4. Training time: ~30-60 minutes depending on hardware")
    print("\n" + "="*70)
    
    response = input("\nPress ENTER to start training (or Ctrl+C to cancel)...")
    
    try:
        model, history, metrics = train_model()
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📊 Final Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print("\n" + "="*70)
        print("✅ Your model is ready!")
        print("="*70)
        print("\n📁 Generated files:")
        print(f"  • Model: p300_eegnet_*.h5")
        print(f"  • Metrics: p300_eegnet_*_metrics.json")
        print(f"  • History: p300_eegnet_*_history.json")
        print(f"  • Plot: p300_eegnet_*_training.png")
        
        if metrics['accuracy'] >= 0.80:
            print("\n✅ EXCELLENT! Model performance is good for deployment.")
        elif metrics['accuracy'] >= 0.70:
            print("\n⚠️  Model is functional but could be improved.")
            print("   Try adjusting undersample_ratio or augmentation_factor.")
        else:
            print("\n❌ Performance is below expected. Check:")
            print("   1. Data quality (run diagnosis script)")
            print("   2. Verify erp_labels are correct")
            print("   3. Try different balance_strategy")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n❌ Training cancelled by user.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        print("TROUBLESHOOTING:")
        print("="*70)
        print("1. Check dataset path is correct")
        print("2. Verify HDF5 file has 'features' and 'erp_labels' keys")
        print("3. Ensure sufficient memory (reduce batch_size if needed)")
        print("4. Try running the diagnosis script first")
        print("="*70)