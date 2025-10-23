"""
BigP3 EEG Signal Visualization and Analysis Tool
Comprehensive plotting and analysis of EEG signals from the BigP3 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import scipy.io
from scipy import signal
from scipy.stats import zscore
import mne
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BigP3SignalPlotter:
    """Comprehensive EEG signal plotting and analysis for BigP3 dataset"""
    
    def __init__(self, data_file_path):
        """
        Initialize the plotter with a BigP3 data file
        
        Args:
            data_file_path: Path to the BigP3 data file
        """
        self.data_file = Path(data_file_path)
        self.raw_data = None
        self.events = None
        self.sample_rate = None
        self.channels = None
        self.subject_id = None
        
        # Load the data
        self.load_data()
        
        # Analysis parameters
        self.epoch_window = (-0.2, 0.8)  # Extended window for better visualization
        self.baseline_window = (-0.2, 0.0)
        self.p300_window = (0.25, 0.45)  # Typical P300 latency window
        
        print(f"📊 Loaded BigP3 data:")
        print(f"   File: {self.data_file.name}")
        print(f"   Shape: {self.raw_data.shape}")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Channels: {len(self.channels)}")
        print(f"   Duration: {self.raw_data.shape[1] / self.sample_rate:.1f} seconds")
        print(f"   Events: {len(self.events) if self.events is not None else 'None'}")
    
    def load_data(self):
        """Load BigP3 data from file"""
        file_ext = self.data_file.suffix.lower()
        
        try:
            if file_ext == '.mat':
                self._load_mat_file()
            elif file_ext == '.edf':
                self._load_edf_file()
            elif file_ext == '.set':
                self._load_set_file()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for demonstration
            self._create_dummy_data()
    
    def _load_mat_file(self):
        """Load MATLAB file"""
        data = scipy.io.loadmat(self.data_file)
        
        # Extract data (adapt based on actual BigP3 structure)
        if 'EEG' in data:
            eeg = data['EEG']
            if hasattr(eeg, 'data'):
                self.raw_data = eeg.data
                self.sample_rate = float(eeg.srate) if hasattr(eeg, 'srate') else 256
                self.events = eeg.event if hasattr(eeg, 'event') else None
            else:
                self.raw_data = eeg
                self.sample_rate = 256
        else:
            # Try common field names
            self.raw_data = data.get('data', data.get('signal', list(data.values())[0]))
            self.sample_rate = data.get('fs', data.get('srate', 256))
            self.events = data.get('events', data.get('triggers', None))
        
        # Set up channels
        self.channels = [f'Ch{i+1}' for i in range(self.raw_data.shape[0])]
        if self.raw_data.shape[0] == 8:
            self.channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']
    
    def _load_edf_file(self):
        """Load EDF file using MNE"""
        raw = mne.io.read_raw_edf(self.data_file, preload=True, verbose=False)
        self.raw_data = raw.get_data()
        self.sample_rate = raw.info['sfreq']
        self.channels = raw.ch_names
        
        # Get events
        try:
            events = mne.find_events(raw, verbose=False)
            self.events = events
        except:
            self.events = None
    
    def _load_set_file(self):
        """Load EEGLAB SET file using MNE"""
        raw = mne.io.read_raw_eeglab(self.data_file, preload=True, verbose=False)
        self.raw_data = raw.get_data()
        self.sample_rate = raw.info['sfreq']
        self.channels = raw.ch_names
        
        # Get events
        try:
            events = mne.find_events(raw, verbose=False)
            self.events = events
        except:
            self.events = None
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration"""
        print("Creating dummy data for demonstration...")
        duration = 300  # 5 minutes
        self.sample_rate = 256
        n_samples = int(duration * self.sample_rate)
        n_channels = 8
        
        # Generate realistic EEG-like data
        np.random.seed(42)
        
        # Base EEG with multiple frequency components
        t = np.arange(n_samples) / self.sample_rate
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Alpha rhythm (8-12 Hz)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
            
            # Beta rhythm (13-30 Hz) 
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
            
            # Theta rhythm (4-8 Hz)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.random() * 2 * np.pi)
            
            # Noise
            noise = np.random.randn(n_samples) * 0.2
            
            # Combine
            eeg_data[ch, :] = alpha + beta + theta + noise
        
        self.raw_data = eeg_data * 1e-6  # Convert to microvolts
        self.channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']
        
        # Create dummy events (P300 speller stimuli)
        n_events = 200
        event_samples = np.sort(np.random.choice(
            range(int(0.5 * self.sample_rate), n_samples - int(0.5 * self.sample_rate)), 
            n_events, replace=False
        ))
        event_codes = np.random.choice([1, 2], n_events)  # 1=target, 2=non-target
        
        self.events = np.column_stack([event_samples, np.zeros(n_events), event_codes])
    
    def plot_comprehensive_overview(self, duration=10.0, start_time=60.0):
        """
        Create a comprehensive overview plot of the EEG signal
        
        Args:
            duration: Duration to plot in seconds
            start_time: Start time in seconds
        """
        print(f"🎨 Creating comprehensive signal overview...")
        
        # Calculate sample indices
        start_idx = int(start_time * self.sample_rate)
        end_idx = int((start_time + duration) * self.sample_rate)
        end_idx = min(end_idx, self.raw_data.shape[1])
        
        # Extract data segment
        data_segment = self.raw_data[:, start_idx:end_idx]
        time_axis = np.arange(start_idx, end_idx) / self.sample_rate
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Raw EEG Signals (Top panel)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_multichannel_eeg(ax1, data_segment, time_axis, title="Raw EEG Signals")
        
        # 2. Power Spectral Density
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_power_spectrum(ax2, data_segment)
        
        # 3. Channel-wise statistics
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_channel_statistics(ax3, data_segment)
        
        # 4. Time-frequency analysis (spectrogram)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_spectrogram(ax4, data_segment, time_axis)
        
        # 5. Signal quality assessment
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_signal_quality(ax5, data_segment)
        
        # 6. Correlation matrix
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_correlation_matrix(ax6, data_segment)
        
        # 7. Amplitude distribution
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_amplitude_distribution(ax7, data_segment)
        
        # 8. Events overlay (if available)
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_events_overlay(ax8, data_segment, time_axis, start_time, duration)
        
        # Overall title and info
        fig.suptitle(f'BigP3 EEG Signal Analysis - {self.data_file.name}\n'
                    f'Time: {start_time:.1f}s - {start_time + duration:.1f}s | '
                    f'Sample Rate: {self.sample_rate} Hz | '
                    f'Channels: {len(self.channels)}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_multichannel_eeg(self, ax, data, time_axis, title="EEG Signals"):
        """Plot multichannel EEG with proper scaling"""
        n_channels = data.shape[0]
        
        # Scale and offset channels for visibility
        scaling_factor = np.std(data) * 3
        offsets = np.arange(n_channels) * scaling_factor
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
        
        for i, (channel_data, offset, color) in enumerate(zip(data, offsets, colors)):
            ax.plot(time_axis, channel_data + offset, color=color, linewidth=0.8, 
                   label=self.channels[i], alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Channels', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(offsets)
        ax.set_yticklabels(self.channels)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add amplitude scale bar
        scale_bar_length = scaling_factor * 0.5
        ax.plot([time_axis[-1] * 0.95, time_axis[-1] * 0.95], 
               [offsets[0] - scale_bar_length/2, offsets[0] + scale_bar_length/2], 
               'k-', linewidth=2)
        ax.text(time_axis[-1] * 0.96, offsets[0], f'{scale_bar_length*1e6:.0f} μV', 
               rotation=90, ha='left', va='center', fontsize=10)
    
    def _plot_power_spectrum(self, ax, data):
        """Plot power spectral density"""
        # Calculate PSD for each channel
        freqs, psd = signal.welch(data, fs=self.sample_rate, nperseg=1024)
        
        # Plot each channel
        colors = plt.cm.tab10(np.linspace(0, 1, data.shape[0]))
        for i, (channel_psd, color) in enumerate(zip(psd, colors)):
            ax.semilogy(freqs, channel_psd, color=color, alpha=0.7, 
                       linewidth=1.5, label=self.channels[i])
        
        # Highlight frequency bands
        alpha_band = (8, 12)
        beta_band = (13, 30)
        theta_band = (4, 8)
        
        ax.axvspan(theta_band[0], theta_band[1], alpha=0.2, color='green', label='Theta (4-8 Hz)')
        ax.axvspan(alpha_band[0], alpha_band[1], alpha=0.2, color='blue', label='Alpha (8-12 Hz)')
        ax.axvspan(beta_band[0], beta_band[1], alpha=0.2, color='red', label='Beta (13-30 Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=12)
        ax.set_title('Power Spectral Density', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_channel_statistics(self, ax, data):
        """Plot channel-wise statistics"""
        # Calculate statistics
        means = np.mean(data, axis=1) * 1e6  # Convert to μV
        stds = np.std(data, axis=1) * 1e6
        mins = np.min(data, axis=1) * 1e6
        maxs = np.max(data, axis=1) * 1e6
        
        x_pos = np.arange(len(self.channels))
        
        # Bar plot with error bars
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(self.channels))))
        
        # Add min/max indicators
        for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
            ax.plot([i, i], [min_val, max_val], 'r-', alpha=0.5, linewidth=2)
            ax.plot(i, min_val, 'rv', markersize=4)
            ax.plot(i, max_val, 'r^', markersize=4)
        
        ax.set_xlabel('Channels', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title('Channel Statistics', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.channels, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(['Min/Max Range', 'Minimum', 'Maximum', 'Mean ± Std'], 
                 loc='upper right', fontsize=8)
    
    def _plot_spectrogram(self, ax, data, time_axis):
        """Plot spectrogram of a representative channel"""
        # Use central parietal channel (Pz) if available, otherwise first channel
        ch_idx = 3 if len(self.channels) >= 4 else 0
        channel_data = data[ch_idx, :]
        
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(channel_data, fs=self.sample_rate, 
                                      nperseg=256, noverlap=128)
        
        # Convert to dB and plot
        Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
        im = ax.pcolormesh(t + time_axis[0], f, Sxx_db, shading='gouraud', 
                          cmap='viridis')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Spectrogram - {self.channels[ch_idx]}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 50)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10)
    
    def _plot_signal_quality(self, ax, data):
        """Plot signal quality metrics"""
        # Calculate quality metrics
        signal_to_noise = []
        kurtosis_vals = []
        
        for ch_data in data:
            # Simple SNR estimate (signal power / noise power)
            signal_power = np.var(ch_data)
            # Estimate noise as high-frequency content
            b, a = signal.butter(4, 30/(self.sample_rate/2), 'high')
            noise = signal.filtfilt(b, a, ch_data)
            noise_power = np.var(noise)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            signal_to_noise.append(snr)
            
            # Kurtosis (measure of outliers/artifacts)
            kurt = scipy.stats.kurtosis(ch_data)
            kurtosis_vals.append(kurt)
        
        # Plot
        x_pos = np.arange(len(self.channels))
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos - 0.2, signal_to_noise, 0.4, label='SNR (dB)', 
                      color='skyblue', alpha=0.7)
        bars2 = ax2.bar(x_pos + 0.2, kurtosis_vals, 0.4, label='Kurtosis', 
                       color='orange', alpha=0.7)
        
        ax.set_xlabel('Channels', fontsize=12)
        ax.set_ylabel('SNR (dB)', fontsize=12, color='blue')
        ax2.set_ylabel('Kurtosis', fontsize=12, color='orange')
        ax.set_title('Signal Quality Assessment', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.channels, rotation=45)
        
        # Add quality thresholds
        ax.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Good SNR threshold')
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Artifact threshold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    
    def _plot_correlation_matrix(self, ax, data):
        """Plot correlation matrix between channels"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add labels
        ax.set_xticks(range(len(self.channels)))
        ax.set_yticks(range(len(self.channels)))
        ax.set_xticklabels(self.channels, rotation=45, ha='right')
        ax.set_yticklabels(self.channels)
        ax.set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=10)
        
        # Add correlation values as text
        for i in range(len(self.channels)):
            for j in range(len(self.channels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                             ha='center', va='center', fontsize=8, 
                             color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    def _plot_amplitude_distribution(self, ax, data):
        """Plot amplitude distribution across channels"""
        # Flatten all data
        all_data = data.flatten() * 1e6  # Convert to μV
        
        # Create histogram
        n, bins, patches = ax.hist(all_data, bins=100, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = scipy.stats.norm.fit(all_data)
        x = np.linspace(all_data.min(), all_data.max(), 100)
        ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal fit (μ={mu:.1f}, σ={sigma:.1f})')
        
        # Add statistics
        ax.axvline(np.mean(all_data), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_data):.1f} μV')
        ax.axvline(np.median(all_data), color='green', linestyle='--', 
                  label=f'Median: {np.median(all_data):.1f} μV')
        
        ax.set_xlabel('Amplitude (μV)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Amplitude Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_events_overlay(self, ax, data, time_axis, start_time, duration):
        """Plot events overlay on EEG signal"""
        if self.events is None:
            ax.text(0.5, 0.5, 'No events available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Events Overlay', fontsize=14, fontweight='bold')
            return
        
        # Plot a representative channel
        ch_idx = 3 if len(self.channels) >= 4 else 0  # Pz or first channel
        channel_data = data[ch_idx, :] * 1e6  # Convert to μV
        
        ax.plot(time_axis, channel_data, 'b-', alpha=0.7, linewidth=1, 
               label=self.channels[ch_idx])
        
        # Find events in the current time window
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        
        # Filter events within time window
        event_mask = ((self.events[:, 0] >= start_sample) & 
                     (self.events[:, 0] <= end_sample))
        windowed_events = self.events[event_mask]
        
        if len(windowed_events) > 0:
            # Plot event markers
            event_times = windowed_events[:, 0] / self.sample_rate
            event_codes = windowed_events[:, 2] if windowed_events.shape[1] > 2 else windowed_events[:, 1]
            
            # Different colors for different event types
            unique_codes = np.unique(event_codes)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_codes)))
            
            for code, color in zip(unique_codes, colors):
                code_mask = event_codes == code
                code_times = event_times[code_mask]
                
                # Plot vertical lines
                for event_time in code_times:
                    ax.axvline(event_time, color=color, alpha=0.7, linewidth=2, 
                              label=f'Event {int(code)}' if code == unique_codes[0] else "")
                
            # Create legend for events
            handles, labels = ax.get_legend_handles_labels()
            event_labels = [f'Event {int(code)}' for code in unique_codes]
            ax.legend(handles[:1] + handles[-len(unique_codes):], 
                     labels[:1] + event_labels, 
                     loc='upper right', fontsize=10)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title(f'Events Overlay - {self.channels[ch_idx]}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_p300_analysis(self, target_events=None, non_target_events=None):
        """
        Create detailed P300 analysis plots
        
        Args:
            target_events: List of event codes for target stimuli
            non_target_events: List of event codes for non-target stimuli
        """
        if self.events is None:
            print("❌ No events available for P300 analysis")
            return None
        
        print("🧠 Creating P300 analysis plots...")
        
        # Default event codes if not provided
        if target_events is None:
            target_events = [1, 3, 5]  # Adapt based on your dataset
        if non_target_events is None:
            non_target_events = [2, 4, 6]
        
        # Extract P300 epochs
        target_epochs, target_times = self._extract_epochs(target_events)
        nontarget_epochs, nontarget_times = self._extract_epochs(non_target_events)
        
        if len(target_epochs) == 0 and len(nontarget_epochs) == 0:
            print("❌ No P300 epochs found")
            return None
        
        # Create comprehensive P300 figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Grand average ERPs
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_grand_average_erp(ax1, target_epochs, nontarget_epochs, target_times)
        
        # 2. Channel-wise ERPs
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_channel_wise_erp(ax2, target_epochs, nontarget_epochs, target_times)
        
        # 3. P300 topography
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_p300_topography(ax3, target_epochs, nontarget_epochs, target_times)
        
        # 4. Single trial visualization
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_single_trials(ax4, target_epochs, nontarget_epochs, target_times)
        
        # 5. Statistical analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_statistical_analysis(ax5, target_epochs, nontarget_epochs, target_times)
        
        # 6. Latency analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_latency_analysis(ax6, target_epochs, nontarget_epochs, target_times)
        
        # 7. Amplitude analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_amplitude_analysis(ax7, target_epochs, nontarget_epochs, target_times)
        
        fig.suptitle(f'P300 Analysis - {self.data_file.name}\n'
                    f'Target trials: {len(target_epochs)} | Non-target trials: {len(nontarget_epochs)}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _extract_epochs(self, event_codes):
        """Extract epochs for specific event codes"""
        if self.events is None:
            return [], []
        
        # Find events matching the codes
        if self.events.shape[1] >= 3:
            event_samples = self.events[:, 0]
            codes = self.events[:, 2]
        else:
            event_samples = self.events[:, 0] 
            codes = self.events[:, 1]
        
        # Filter for target events
        target_mask = np.isin(codes, event_codes)
        target_samples = event_samples[target_mask]
        
        # Extract epochs
        epoch_start_samples = int(self.epoch_window[0] * self.sample_rate)
        epoch_end_samples = int(self.epoch_window[1] * self.sample_rate)
        epoch_length = epoch_end_samples - epoch_start_samples
        
        epochs = []
        for sample in target_samples:
            start_idx = int(sample + epoch_start_samples)
            end_idx = int(sample + epoch_end_samples)
            
            # Check bounds
            if start_idx >= 0 and end_idx < self.raw_data.shape[1]:
                epoch = self.raw_data[:, start_idx:end_idx]
                epochs.append(epoch)
        
        epochs = np.array(epochs) if epochs else np.array([]).reshape(0, self.raw_data.shape[0], epoch_length)
        
        # Create time axis for epochs
        time_axis = np.linspace(self.epoch_window[0], self.epoch_window[1], epoch_length)
        
        return epochs, time_axis
    
    def _plot_grand_average_erp(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot grand average ERP comparison"""
        if len(target_epochs) > 0:
            # Calculate grand average (across channels and trials)
            target_ga = np.mean(np.mean(target_epochs, axis=0), axis=0) * 1e6
            target_sem = scipy.stats.sem(np.mean(target_epochs, axis=1).flatten()) * 1e6
            
            ax.plot(time_axis, target_ga, 'r-', linewidth=3, label=f'Target (n={len(target_epochs)})')
            ax.fill_between(time_axis, target_ga - target_sem, target_ga + target_sem, 
                           alpha=0.3, color='red')
        
        if len(nontarget_epochs) > 0:
            nontarget_ga = np.mean(np.mean(nontarget_epochs, axis=0), axis=0) * 1e6
            nontarget_sem = scipy.stats.sem(np.mean(nontarget_epochs, axis=1).flatten()) * 1e6
            
            ax.plot(time_axis, nontarget_ga, 'b-', linewidth=3, label=f'Non-target (n={len(nontarget_epochs)})')
            ax.fill_between(time_axis, nontarget_ga - nontarget_sem, nontarget_ga + nontarget_sem, 
                           alpha=0.3, color='blue')
        
        # Mark important time points
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
        ax.axvspan(self.p300_window[0], self.p300_window[1], alpha=0.2, color='yellow', 
                  label='P300 window')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title('Grand Average Event-Related Potentials', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_channel_wise_erp(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot ERPs for each channel separately"""
        n_channels = len(self.channels)
        
        # Calculate scaling for display
        if len(target_epochs) > 0:
            all_data = np.concatenate([target_epochs.flatten(), nontarget_epochs.flatten()])
            scale = np.std(all_data) * 1e6 * 2
        else:
            scale = 10  # Default scale
        
        colors_target = plt.cm.Reds(np.linspace(0.4, 1, n_channels))
        colors_nontarget = plt.cm.Blues(np.linspace(0.4, 1, n_channels))
        
        for ch in range(n_channels):
            offset = ch * scale
            
            if len(target_epochs) > 0:
                target_ch = np.mean(target_epochs[:, ch, :], axis=0) * 1e6
                ax.plot(time_axis, target_ch + offset, color=colors_target[ch], 
                       linewidth=2, alpha=0.8)
            
            if len(nontarget_epochs) > 0:
                nontarget_ch = np.mean(nontarget_epochs[:, ch, :], axis=0) * 1e6
                ax.plot(time_axis, nontarget_ch + offset, color=colors_nontarget[ch], 
                       linewidth=2, alpha=0.8, linestyle='--')
        
        # Add channel labels
        offsets = np.arange(n_channels) * scale
        ax.set_yticks(offsets)
        ax.set_yticklabels(self.channels)
        
        # Add stimulus onset marker
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Channels', fontsize=12)
        ax.set_title('Channel-wise ERPs\n(Solid: Target, Dashed: Non-target)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_p300_topography(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot P300 topography (simplified for 8-channel setup)"""
        if len(target_epochs) == 0:
            ax.text(0.5, 0.5, 'No target epochs available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('P300 Topography', fontsize=14, fontweight='bold')
            return
        
        # Calculate P300 amplitude (average in P300 window)
        p300_start_idx = np.argmin(np.abs(time_axis - self.p300_window[0]))
        p300_end_idx = np.argmin(np.abs(time_axis - self.p300_window[1]))
        
        target_p300 = np.mean(target_epochs[:, :, p300_start_idx:p300_end_idx], axis=(0, 2)) * 1e6
        
        if len(nontarget_epochs) > 0:
            nontarget_p300 = np.mean(nontarget_epochs[:, :, p300_start_idx:p300_end_idx], axis=(0, 2)) * 1e6
            p300_diff = target_p300 - nontarget_p300
        else:
            p300_diff = target_p300
        
        # Simple topography for 8-channel montage
        # Approximate electrode positions (normalized)
        positions = {
            'Fz': (0.5, 0.8),   'Cz': (0.5, 0.5),   'P3': (0.2, 0.3),
            'Pz': (0.5, 0.2),   'P4': (0.8, 0.3),   'PO7': (0.2, 0.1),
            'Oz': (0.5, 0.0),   'PO8': (0.8, 0.1)
        }
        
        # Create scatter plot with color-coded amplitudes
        x_pos = [positions.get(ch, (0.5, 0.5))[0] for ch in self.channels]
        y_pos = [positions.get(ch, (0.5, 0.5))[1] for ch in self.channels]
        
        scatter = ax.scatter(x_pos, y_pos, c=p300_diff[:len(self.channels)], 
                           s=200, cmap='RdBu_r', edgecolors='black', linewidth=2)
        
        # Add channel labels
        for i, ch in enumerate(self.channels):
            if i < len(x_pos):
                ax.annotate(ch, (x_pos[i], y_pos[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add head outline
        head_circle = plt.Circle((0.5, 0.4), 0.45, fill=False, linewidth=2, color='black')
        ax.add_patch(head_circle)
        
        # Add nose
        nose = patches.Polygon([(0.48, 0.85), (0.5, 0.9), (0.52, 0.85)], 
                              closed=True, fill=True, color='black')
        ax.add_patch(nose)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('P300 Topography\n(Target - Non-target)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Amplitude Difference (μV)', fontsize=10)
    
    def _plot_single_trials(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot single trial variability"""
        if len(target_epochs) == 0:
            ax.text(0.5, 0.5, 'No epochs available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('Single Trial Analysis', fontsize=14, fontweight='bold')
            return
        
        # Use Pz channel (index 3) or first available
        ch_idx = 3 if len(self.channels) > 3 else 0
        
        # Plot first few target trials
        n_trials_plot = min(10, len(target_epochs))
        for i in range(n_trials_plot):
            alpha_val = 0.7 - (i * 0.05)  # Fade older trials
            ax.plot(time_axis, target_epochs[i, ch_idx, :] * 1e6, 
                   'r-', alpha=alpha_val, linewidth=1)
        
        # Plot average
        if len(target_epochs) > 0:
            target_avg = np.mean(target_epochs[:, ch_idx, :], axis=0) * 1e6
            ax.plot(time_axis, target_avg, 'r-', linewidth=3, label='Target Average')
        
        if len(nontarget_epochs) > 0:
            nontarget_avg = np.mean(nontarget_epochs[:, ch_idx, :], axis=0) * 1e6
            ax.plot(time_axis, nontarget_avg, 'b-', linewidth=3, label='Non-target Average')
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
        ax.axvspan(self.p300_window[0], self.p300_window[1], alpha=0.2, color='yellow')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title(f'Single Trials - {self.channels[ch_idx]}\n(Showing first {n_trials_plot} trials)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_analysis(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot statistical comparison between conditions"""
        if len(target_epochs) == 0 or len(nontarget_epochs) == 0:
            ax.text(0.5, 0.5, 'Need both target and non-target epochs', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Statistical Analysis', fontsize=14, fontweight='bold')
            return
        
        # Perform t-tests at each time point (across all channels)
        target_data = np.mean(target_epochs, axis=1)  # Average across channels
        nontarget_data = np.mean(nontarget_epochs, axis=1)
        
        t_values = []
        p_values = []
        
        for t_idx in range(target_data.shape[1]):
            t_stat, p_val = scipy.stats.ttest_ind(target_data[:, t_idx], 
                                                 nontarget_data[:, t_idx])
            t_values.append(t_stat)
            p_values.append(p_val)
        
        t_values = np.array(t_values)
        p_values = np.array(p_values)
        
        # Plot t-values
        ax.plot(time_axis, t_values, 'g-', linewidth=2, label='t-statistic')
        
        # Mark significant time points (p < 0.05)
        sig_mask = p_values < 0.05
        if np.any(sig_mask):
            ax.fill_between(time_axis, 0, t_values, where=sig_mask, 
                           alpha=0.3, color='red', label='p < 0.05')
        
        # Add significance thresholds
        t_thresh = scipy.stats.t.ppf(0.975, len(target_data) + len(nontarget_data) - 2)
        ax.axhline(t_thresh, color='r', linestyle='--', alpha=0.7, label=f't-threshold (±{t_thresh:.2f})')
        ax.axhline(-t_thresh, color='r', linestyle='--', alpha=0.7)
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('t-statistic', fontsize=12)
        ax.set_title('Statistical Comparison\n(Target vs Non-target)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_latency_analysis(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot P300 latency analysis"""
        if len(target_epochs) == 0:
            ax.text(0.5, 0.5, 'No target epochs available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('P300 Latency Analysis', fontsize=14, fontweight='bold')
            return
        
        # Find P300 peak latencies for each trial
        p300_start_idx = np.argmin(np.abs(time_axis - self.p300_window[0]))
        p300_end_idx = np.argmin(np.abs(time_axis - self.p300_window[1]))
        
        latencies = []
        
        for trial in target_epochs:
            # Average across channels for each trial
            trial_avg = np.mean(trial, axis=0)
            
            # Find peak in P300 window
            p300_segment = trial_avg[p300_start_idx:p300_end_idx]
            peak_idx = np.argmax(p300_segment) + p300_start_idx
            latency = time_axis[peak_idx] * 1000  # Convert to ms
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        # Plot histogram of latencies
        n, bins, patches = ax.hist(latencies, bins=20, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = scipy.stats.norm.fit(latencies)
        x = np.linspace(latencies.min(), latencies.max(), 100)
        ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal fit\nμ={mu:.1f}ms, σ={sigma:.1f}ms')
        
        # Add statistics
        ax.axvline(np.mean(latencies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(latencies):.1f}ms')
        ax.axvline(np.median(latencies), color='green', linestyle='--', 
                  label=f'Median: {np.median(latencies):.1f}ms')
        
        ax.set_xlabel('P300 Latency (ms)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('P300 Latency Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_amplitude_analysis(self, ax, target_epochs, nontarget_epochs, time_axis):
        """Plot P300 amplitude analysis"""
        if len(target_epochs) == 0:
            ax.text(0.5, 0.5, 'No target epochs available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('P300 Amplitude Analysis', fontsize=14, fontweight='bold')
            return
        
        # Calculate P300 amplitudes
        p300_start_idx = np.argmin(np.abs(time_axis - self.p300_window[0]))
        p300_end_idx = np.argmin(np.abs(time_axis - self.p300_window[1]))
        
        # Target amplitudes
        target_amplitudes = []
        for trial in target_epochs:
            trial_avg = np.mean(trial, axis=0)  # Average across channels
            p300_amp = np.mean(trial_avg[p300_start_idx:p300_end_idx]) * 1e6
            target_amplitudes.append(p300_amp)
        
        # Non-target amplitudes
        nontarget_amplitudes = []
        if len(nontarget_epochs) > 0:
            for trial in nontarget_epochs:
                trial_avg = np.mean(trial, axis=0)
                p300_amp = np.mean(trial_avg[p300_start_idx:p300_end_idx]) * 1e6
                nontarget_amplitudes.append(p300_amp)
        
        # Create box plot
        data_to_plot = []
        labels = []
        
        if target_amplitudes:
            data_to_plot.append(target_amplitudes)
            labels.append(f'Target\n(n={len(target_amplitudes)})')
        
        if nontarget_amplitudes:
            data_to_plot.append(nontarget_amplitudes)
            labels.append(f'Non-target\n(n={len(nontarget_amplitudes)})')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightcoral', 'lightblue']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Add statistical test if both conditions available
        if len(data_to_plot) == 2:
            t_stat, p_val = scipy.stats.ttest_ind(target_amplitudes, nontarget_amplitudes)
            ax.text(0.5, 0.95, f't-test: t={t_stat:.2f}, p={p_val:.4f}', 
                   transform=ax.transAxes, ha='center', va='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel('P300 Amplitude (μV)', fontsize=12)
        ax.set_title('P300 Amplitude Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def save_plots(self, output_dir="plots/", prefix="bigp3_analysis"):
        """Save all generated plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving plots to {output_path}...")
        
        # Generate and save comprehensive overview
        fig1 = self.plot_comprehensive_overview()
        fig1.savefig(output_path / f"{prefix}_overview.png", dpi=300, bbox_inches='tight')
        fig1.savefig(output_path / f"{prefix}_overview.pdf", bbox_inches='tight')
        
        # Generate and save P300 analysis
        fig2 = self.plot_p300_analysis()
        if fig2:
            fig2.savefig(output_path / f"{prefix}_p300.png", dpi=300, bbox_inches='tight')
            fig2.savefig(output_path / f"{prefix}_p300.pdf", bbox_inches='tight')
        
        print(f"✅ Plots saved successfully!")
        
        # Create summary report
        self._create_analysis_report(output_path / f"{prefix}_report.txt")
    
    def _create_analysis_report(self, report_path):
        """Create a text summary of the analysis"""
        with open(report_path, 'w') as f:
            f.write("BigP3 EEG Signal Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Dataset: {self.data_file.name}\n")
            f.write(f"Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"Channels: {len(self.channels)} ({', '.join(self.channels)})\n")
            f.write(f"Duration: {self.raw_data.shape[1] / self.sample_rate:.1f} seconds\n")
            f.write(f"Data Shape: {self.raw_data.shape}\n\n")
            
            if self.events is not None:
                f.write(f"Events: {len(self.events)} total\n")
                unique_events = np.unique(self.events[:, -1])
                f.write(f"Event Types: {unique_events}\n\n")
            else:
                f.write("Events: None available\n\n")
            
            # Signal quality assessment
            signal_quality = np.std(self.raw_data, axis=1)
            f.write("Signal Quality (Standard Deviation by Channel):\n")
            for ch, quality in zip(self.channels, signal_quality):
                f.write(f"  {ch}: {quality*1e6:.2f} μV\n")
            
            f.write(f"\nAnalysis completed: {pd.Timestamp.now()}\n")

def main():
    """Main function for signal plotting"""
    parser = argparse.ArgumentParser(description="Plot and analyze BigP3 EEG signals")
    parser.add_argument("data_file", help="Path to BigP3 data file")
    parser.add_argument("--output", default="plots/", help="Output directory for plots")
    parser.add_argument("--duration", type=float, default=10.0, 
                      help="Duration to plot in seconds")
    parser.add_argument("--start-time", type=float, default=60.0,
                      help="Start time for plotting in seconds")
    parser.add_argument("--save-plots", action='store_true',
                      help="Save plots to files")
    parser.add_argument("--p300-analysis", action='store_true',
                      help="Include P300 analysis")
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = BigP3SignalPlotter(args.data_file)
    
    # Create comprehensive overview
    fig1 = plotter.plot_comprehensive_overview(
        duration=args.duration, 
        start_time=args.start_time
    )
    
    # Create P300 analysis if requested
    if args.p300_analysis:
        fig2 = plotter.plot_p300_analysis()
    
    # Save plots if requested
    if args.save_plots:
        plotter.save_plots(args.output)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    # Example usage if no command line arguments
    import sys
    if len(sys.argv) == 1:
        print("BigP3 Signal Plotter - Example Usage")
        print("="*50)
        
        # Create demonstration with dummy data
        print("Creating demonstration with simulated data...")
        plotter = BigP3SignalPlotter("dummy_data.mat")  # This will create dummy data
        
        # Generate plots
        print("Generating comprehensive overview...")
        fig1 = plotter.plot_comprehensive_overview(duration=10, start_time=30)
        
        print("Generating P300 analysis...")
        fig2 = plotter.plot_p300_analysis()
        
        print("Plots generated! Close the plot windows to continue.")
        plt.show()
        
        print("\nTo use with real data:")
        print("python bigp3_signal_plotter.py /path/to/your/bigp3/file.mat")
        print("python bigp3_signal_plotter.py /path/to/your/bigp3/file.edf --p300-analysis --save-plots")
    else:
        main()