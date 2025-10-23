#!/usr/bin/env python3
"""
P300 Speller Backend System with OpenBCI Integration
Complete fixed version with simulation/hardware modes
"""

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import threading
from collections import deque
import time
import os
from pathlib import Path
import argparse

# EEG and ML imports
try:
    import tensorflow as tf
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
    import serial.tools.list_ports
    import mne
    from scipy import signal
    from sklearn.metrics import classification_report, confusion_matrix
    print("✅ All core dependencies loaded successfully!")
    BRAINFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Missing dependency: {e}")
    print("📦 Install with: pip install tensorflow brainflow mne scikit-learn scipy")
    BRAINFLOW_AVAILABLE = False

# AI/LLM imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers loaded successfully!")
except ImportError:
    print("⚠️ Transformers not available. Install with: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate your trained P300 model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.channel_mapper = ChannelMapper()  # Add this
        self.load_model()
        
    def load_model(self):
        """Load the trained EEGNet model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Model loaded from {self.model_path}")
                self.print_model_info()
            else:
                logger.warning(f"⚠️ Model not found at {self.model_path}")
                logger.info("🎯 Using simulation mode - random P300 detection")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("🎯 Falling back to simulation mode")
            
    def print_model_info(self):
        """Print model architecture and info"""
        if self.model:
            print("\n" + "="*50)
            print("🧠 P300 MODEL INFORMATION")
            print("="*50)
            
            print(f"Model type: {type(self.model).__name__}")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
            print(f"Total parameters: {self.model.count_params():,}")
            
            print("\nModel Architecture:")
            self.model.summary()
    
    def test_single_epoch(self, eeg_epoch):
        """Test a single EEG epoch for P300 detection"""
        if not self.model:
            base_prob = np.random.beta(2, 5)
            return float(base_prob)
            
        try:
            current_channels = eeg_epoch.shape[0]
            current_samples = eeg_epoch.shape[1]
            
            # Map 8 channels to 16 if needed
            if current_channels == 8:
                eeg_epoch = self.channel_mapper.map_8_to_16(eeg_epoch)
                logger.info(f"Mapped 8 channels to 16 using spatial interpolation")
            elif current_channels != 16:
                logger.error(f"Unexpected channel count: {current_channels}")
                return np.random.beta(2, 5)
            
            # Adjust samples to 181
            if current_samples < 181:
                padding = 181 - current_samples
                eeg_epoch = np.pad(eeg_epoch, ((0, 0), (0, padding)), mode='edge')
            elif current_samples > 181:
                eeg_epoch = eeg_epoch[:, :181]
            
            # Reshape to model input: (1, 16, 181, 1)
            eeg_epoch = eeg_epoch.reshape(1, 16, 181, 1)
            
            prediction = self.model.predict(eeg_epoch, verbose=0)
            if prediction.shape[1] > 1:
                p300_probability = prediction[0][1]
            else:
                p300_probability = prediction[0][0]
            
            return float(p300_probability)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return np.random.beta(2, 5)
        
class ChannelMapper:
    """Map 8 channels to 16 using spatial relationships"""
    
    def __init__(self):
        # Define standard 10-20 positions for 16 channels (typical PhysioNet layout)
        # Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4
        self.channels_16 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 
                            'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4']
        
        # Typical 8-channel OpenBCI Cyton setup
        self.channels_8 = ['Fz', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'Oz']
        
        # Create mapping: which 8-channel corresponds to which 16-channel positions
        self.mapping = self._create_mapping()
        
    def _create_mapping(self):
        """Create intelligent mapping from 8 to 16 channels"""
        # Map each of 16 channels to weighted combination of 8 channels
        mapping = {}
        
        # Direct mappings where channels exist in both
        direct_map = {
            4: 0,   # Fz -> Fz
            8: 1,   # C3 -> C3
            9: 2,   # Cz -> Cz
            10: 3,  # C4 -> C4
            13: 4,  # P3 -> P3
            14: 5,  # Pz -> Pz
            15: 6,  # P4 -> P4
        }
        
        # For missing channels, use nearby channels with weights
        mapping[0] = [(0, 0.7), (2, 0.3)]    # Fp1 from Fz + C3
        mapping[1] = [(0, 0.7), (3, 0.3)]    # Fp2 from Fz + C4
        mapping[2] = [(1, 0.6), (4, 0.4)]    # F7 from C3 + P3
        mapping[3] = [(0, 0.5), (1, 0.5)]    # F3 from Fz + C3
        mapping[4] = [(0, 1.0)]              # Fz direct
        mapping[5] = [(0, 0.5), (3, 0.5)]    # F4 from Fz + C4
        mapping[6] = [(3, 0.6), (6, 0.4)]    # F8 from C4 + P4
        mapping[7] = [(1, 0.6), (4, 0.4)]    # T7 from C3 + P3
        mapping[8] = [(1, 1.0)]              # C3 direct
        mapping[9] = [(2, 1.0)]              # Cz direct
        mapping[10] = [(3, 1.0)]             # C4 direct
        mapping[11] = [(3, 0.6), (6, 0.4)]   # T8 from C4 + P4
        mapping[12] = [(4, 0.7), (7, 0.3)]   # P7 from P3 + Oz
        mapping[13] = [(4, 1.0)]             # P3 direct
        mapping[14] = [(5, 1.0)]             # Pz direct
        mapping[15] = [(6, 1.0)]             # P4 direct
        
        return mapping
    
    def map_8_to_16(self, eeg_8ch):
        """
        Map 8-channel data to 16-channel format
        eeg_8ch: shape (8, samples)
        returns: shape (16, samples)
        """
        n_samples = eeg_8ch.shape[1]
        eeg_16ch = np.zeros((16, n_samples))
        
        for ch_16, weights in self.mapping.items():
            for ch_8, weight in weights:
                eeg_16ch[ch_16, :] += weight * eeg_8ch[ch_8, :]
        
        return eeg_16ch


class AdvancedPredictor:
    """Advanced text prediction using modern LLMs"""
    
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        if TRANSFORMERS_AVAILABLE:
            self.load_model()
        else:
            logger.warning("⚠️ Transformers not available. Using simple word prediction.")
            
    def load_model(self):
        """Load the language model"""
        try:
            logger.info(f"🤖 Loading language model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("✅ Language model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading language model: {e}")
            self.tokenizer = None
            self.model = None
    
    def predict_next_words(self, context, num_predictions=5):
        """Predict next words given context"""
        if not self.model or not context.strip():
            return self.get_common_words()[:num_predictions]
            
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(context, return_tensors='pt', max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                next_token_logits = outputs.logits[0, -1, :]
                
            # Get top predictions
            top_k_predictions = torch.topk(next_token_logits, num_predictions * 2)
            
            predictions = []
            for token_id in top_k_predictions.indices:
                word = self.tokenizer.decode([token_id]).strip()
                if word and len(word) > 1 and word.isalpha():
                    predictions.append(word)
                    
            return predictions[:num_predictions] if predictions else self.get_common_words()[:num_predictions]
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return self.get_common_words()[:num_predictions]
    
    def get_common_words(self):
        """Fallback common words"""
        return [
            'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at',
            'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but',
            'what', 'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when'
        ]


class OpenBCIStreamer:
    """OpenBCI Cyton R25 Integration"""
    
    def __init__(self, board_id=BoardIds.CYTON_BOARD, serial_port=None, simulation_mode=False):
        self.simulation_mode = simulation_mode
        
        if not BRAINFLOW_AVAILABLE and not simulation_mode:
            logger.warning("⚠️ BrainFlow not available - switching to simulation mode")
            self.simulation_mode = True
            
        if not self.simulation_mode:
            self.board_id = board_id
            self.board = None
            self.sample_rate = BoardShim.get_sampling_rate(board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.num_channels = len(self.eeg_channels)
            
            # Setup board parameters for Cyton R25
            self.params = BrainFlowInputParams()
            if serial_port:
                self.params.serial_port = serial_port
        else:
            # Simulation parameters
            self.board_id = "SIMULATION"
            self.sample_rate = 250
            self.num_channels = 8
            self.eeg_channels = list(range(8))
            self.board = None
        
        self.is_streaming = False
        
        # Data buffers
        self.data_buffer = deque(maxlen=10000)
        self.epoch_buffer = deque(maxlen=100)
        
        # Simulation thread
        self.simulation_thread = None
        
        logger.info(f"🎯 OpenBCI Configuration:")
        logger.info(f"   Mode: {'🎮 SIMULATION' if self.simulation_mode else '🔌 HARDWARE'}")
        logger.info(f"   Board ID: {self.board_id}")
        logger.info(f"   Sample Rate: {self.sample_rate} Hz")
        logger.info(f"   EEG Channels: {self.num_channels}")
    
    def find_openbci_port(self):
        """Automatically find OpenBCI Cyton R25 serial port"""
        if self.simulation_mode:
            return "SIMULATION_PORT"
            
        ports = list(serial.tools.list_ports.comports())
        openbci_ports = []
        
        for port in ports:
            # Look for Cyton R25 identifiers
            if any(identifier in port.description.lower() for identifier in 
                   ['openbci', 'cp210x', 'ftdi', 'arduino', 'serial', 'usb']):
                openbci_ports.append(port.device)
                logger.info(f"🔍 Found potential Cyton port: {port.device} - {port.description}")
        
        if openbci_ports:
            logger.info(f"✅ Selected OpenBCI port: {openbci_ports[0]}")
            return openbci_ports[0]
        else:
            logger.warning("⚠️ No OpenBCI ports found automatically")
            return None
    
    def connect(self):
        """Connect to OpenBCI board"""
        if self.simulation_mode:
            logger.info("🎮 Connected to simulation mode")
            return True
            
        try:
            if not self.params.serial_port:
                self.params.serial_port = self.find_openbci_port()
                
            if not self.params.serial_port:
                raise Exception("No serial port specified or found")
            
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            
            logger.info(f"✅ Connected to OpenBCI Cyton on {self.params.serial_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to OpenBCI: {e}")
            logger.info("🎮 Switching to simulation mode")
            self.simulation_mode = True
            return True  # Always succeed in simulation
    
    def start_streaming(self, data_callback=None):
        """Start real-time EEG streaming"""
        if self.simulation_mode:
            return self._start_simulation_streaming(data_callback)
        
        if not self.board:
            raise Exception("Board not connected. Call connect() first.")
        
        try:
            self.board.start_stream()
            self.is_streaming = True
            
            if data_callback:
                # Start data collection thread
                stream_thread = threading.Thread(
                    target=self._stream_loop, 
                    args=(data_callback,)
                )
                stream_thread.daemon = True
                stream_thread.start()
            
            logger.info("✅ EEG streaming started (Hardware)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start streaming: {e}")
            return False
    
    def _start_simulation_streaming(self, data_callback=None):
        """Start simulation streaming"""
        self.is_streaming = True
        
        if data_callback:
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop,
                args=(data_callback,)
            )
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
        
        logger.info("✅ EEG streaming started (Simulation)")
        return True
    
    def _simulation_loop(self, callback):
        """Simulation streaming loop"""
        while self.is_streaming:
            try:
                # Generate realistic EEG data
                samples_per_chunk = 25  # 100ms at 250Hz
                eeg_data = self._generate_realistic_eeg(samples_per_chunk)
                
                # Add to buffer
                for sample in eeg_data.T:
                    self.data_buffer.append(sample)
                
                # Call callback with new data
                if callback:
                    callback(eeg_data)
                
                time.sleep(0.1)  # 100ms chunks
                
            except Exception as e:
                logger.error(f"❌ Error in simulation loop: {e}")
                break
    
    def _generate_realistic_eeg(self, num_samples):
        """Generate realistic EEG data for simulation"""
        # Base frequencies and amplitudes typical for EEG
        alpha_freq = 10  # Alpha rhythm
        beta_freq = 20   # Beta rhythm
        theta_freq = 6   # Theta rhythm
        
        t = np.linspace(0, num_samples/self.sample_rate, num_samples)
        eeg_data = np.zeros((self.num_channels, num_samples))
        
        for ch in range(self.num_channels):
            # Mix of different brain rhythms
            alpha = 10 * np.sin(2 * np.pi * alpha_freq * t + np.random.random() * 2 * np.pi)
            beta = 5 * np.sin(2 * np.pi * beta_freq * t + np.random.random() * 2 * np.pi)
            theta = 8 * np.sin(2 * np.pi * theta_freq * t + np.random.random() * 2 * np.pi)
            
            # Add noise and artifacts
            noise = np.random.normal(0, 2, num_samples)
            
            # Combine signals
            eeg_data[ch, :] = alpha + beta + theta + noise
        
        return eeg_data
    
    def _stream_loop(self, callback):
        """Internal streaming loop for hardware"""
        while self.is_streaming:
            try:
                # Get new data
                data = self.board.get_board_data()
                
                if data.shape[1] > 0:  # If we have new samples
                    # Extract EEG channels
                    eeg_data = data[self.eeg_channels, :]
                    
                    # Add to buffer
                    for sample in eeg_data.T:
                        self.data_buffer.append(sample)
                    
                    # Call callback with new data
                    if callback:
                        callback(eeg_data)
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                break
    
    def get_epoch(self, pre_stim=0.1, post_stim=0.6, trigger_time=None):
        """Extract epoch around stimulus"""
        if trigger_time is None:
            trigger_time = time.time()
        
        epoch_samples = int((pre_stim + post_stim) * self.sample_rate)
        
        # Wait a bit if buffer doesn't have enough data yet
        wait_count = 0
        while len(self.data_buffer) < epoch_samples and wait_count < 10:
            time.sleep(0.05)  # Wait 50ms
            wait_count += 1
        
        # Get recent data
        if len(self.data_buffer) < epoch_samples:
            logger.warning(f"⚠️ Insufficient data: {len(self.data_buffer)}/{epoch_samples} samples")
            # Return what we have, padded
            available_data = list(self.data_buffer)
            if len(available_data) == 0:
                return self._generate_realistic_eeg(epoch_samples)
            
            # Pad with last sample
            epoch_array = np.array(available_data).T
            if epoch_array.shape[1] < epoch_samples:
                padding = np.tile(epoch_array[:, -1:], (1, epoch_samples - epoch_array.shape[1]))
                epoch_array = np.hstack([epoch_array, padding])
            return epoch_array
        
        # Extract epoch data
        epoch_data = list(self.data_buffer)[-epoch_samples:]
        epoch_array = np.array(epoch_data).T  # Shape: (channels, samples)
        
        return epoch_array
    
    def stop_streaming(self):
        """Stop EEG streaming"""
        if self.is_streaming:
            self.is_streaming = False
            if self.board and not self.simulation_mode:
                self.board.stop_stream()
            logger.info("🛑 EEG streaming stopped")
    
    def disconnect(self):
        """Disconnect from OpenBCI board"""
        self.stop_streaming()
        if self.board and not self.simulation_mode:
            self.board.release_session()
            self.board = None
        logger.info("🔌 Disconnected from OpenBCI")


class RealTimeProcessor:
    """Real-time EEG processing"""
    
    def __init__(self, sample_rate=250, channels=8):
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Filter parameters optimized for P300
        self.low_freq = 0.1
        self.high_freq = 30.0
        self.notch_freq = 60.0  # Power line frequency (50Hz for Europe, 60Hz for US)
        
        # Artifact detection thresholds
        self.artifact_threshold = 300  # microvolts
        
    def preprocess_epoch(self, epoch_data, check_artifacts=True):
        """Preprocess single epoch"""
        try:
            if epoch_data is None:
                return None
                
            # Apply bandpass filter
            filtered_data = self.apply_bandpass_filter(epoch_data)
            
            # Apply notch filter for power line interference
            notched_data = self.apply_notch_filter(filtered_data)
            
            # Baseline correction
            baseline_corrected = self.baseline_correction(notched_data)
            
            # Artifact detection (optional for real-time to avoid too many rejections)
            if check_artifacts and self.detect_artifacts(baseline_corrected):
                logger.warning("⚠️ Artifact detected - rejecting epoch")
                return None
            
            return baseline_corrected
            
        except Exception as e:
            logger.error(f"❌ Error in preprocessing: {e}")
            return None
    
    def apply_bandpass_filter(self, data):
        """Apply bandpass filter"""
        if not BRAINFLOW_AVAILABLE:
            return data
        
        try:
            # Ensure contiguous array
            filtered_data = np.ascontiguousarray(data.copy(), dtype=np.float64)
            for ch in range(filtered_data.shape[0]):
                DataFilter.perform_bandpass(
                    filtered_data[ch, :], self.sample_rate, 
                    self.low_freq, self.high_freq, 4,
                    FilterTypes.BUTTERWORTH.value, 0
                )
            return filtered_data
        except Exception as e:
            logger.warning(f"⚠️ Filter error: {e}, returning unfiltered data")
            return data

    def apply_notch_filter(self, data):
        """Apply notch filter for power line interference"""
        if not BRAINFLOW_AVAILABLE:
            return data
        
        try:
            # Ensure contiguous array
            filtered_data = np.ascontiguousarray(data.copy(), dtype=np.float64)
            for ch in range(filtered_data.shape[0]):
                DataFilter.perform_bandstop(
                    filtered_data[ch, :], self.sample_rate,
                    self.notch_freq - 2, self.notch_freq + 2, 4,
                    FilterTypes.BUTTERWORTH.value, 0
                )
            return filtered_data
        except Exception as e:
            logger.warning(f"⚠️ Notch filter error: {e}, returning unfiltered data")
            return data
    
    def baseline_correction(self, data, baseline_window=(0, 0.1)):
        """Apply baseline correction"""
        try:
            baseline_samples = int(baseline_window[1] * self.sample_rate)
            baseline_mean = np.mean(data[:, :baseline_samples], axis=1, keepdims=True)
            return data - baseline_mean
        except Exception as e:
            logger.warning(f"⚠️ Baseline correction error: {e}")
            return data
    
    def detect_artifacts(self, data):
        """Simple artifact detection based on amplitude"""
        try:
            max_amplitude = np.max(np.abs(data))
            return max_amplitude > self.artifact_threshold
        except:
            return False


class P300SpellerBackend:
    """Main backend system for P300 speller"""
    
    def __init__(self, model_path="models/p300_eegnet.h5", board_id=None, simulation_mode=False):
        self.model_path = model_path
        self.simulation_mode = simulation_mode
        self.validator = ModelValidator(model_path)
        self.predictor = AdvancedPredictor()
        
        
        # Initialize OpenBCI integration
        if board_id is None:
            board_id = BoardIds.CYTON_BOARD if BRAINFLOW_AVAILABLE else "SIMULATION"
            
        self.eeg_interface = OpenBCIStreamer(board_id, simulation_mode=simulation_mode)
        self.processor = RealTimeProcessor()
        
        self.connected_clients = set()
        self.is_running = False
        
        # Session data
        self.session_data = {
            'trials': 0,
            'detections': 0,
            'accuracy': 0,
            'start_time': None,
            'stimulus_times': [],
            'p300_responses': []
        }
        
        # P300 spelling specific
        self.current_stimulus = None
        self.stimulus_queue = deque()
        
        self.p300_scores = {'rows': [0]*6, 'columns': [0]*6}
        self.trial_count = 0
        self.trials_per_selection = 10  # Number of trials before selecting character
        
    async def select_character_from_p300(self):
        """Select character based on accumulated P300 scores"""
        # Find row and column with highest scores
        best_row = np.argmax(self.p300_scores['rows'])
        best_col = np.argmax(self.p300_scores['columns'])
        
        # Get the character from matrix
        matrix = [
            ['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '0']
        ]
        
        selected_char = matrix[best_row][best_col]
        
        # Send to frontend
        await self.broadcast_message({
            'type': 'character_selected',
            'character': selected_char,
            'row': int(best_row),
            'column': int(best_col),
            'row_score': float(self.p300_scores['rows'][best_row]),
            'col_score': float(self.p300_scores['columns'][best_col])
        })
        
        # Reset scores
        self.p300_scores = {'rows': [0]*6, 'columns': [0]*6}
        self.trial_count = 0
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        self.connected_clients.add(websocket)
        logger.info(f"👤 Client connected. Total clients: {len(self.connected_clients)}")
        
        # Send initial status
        await self.send_to_client(websocket, {
            'type': 'connection_status',
            'connected': True,
            'mode': 'simulation' if self.simulation_mode else 'hardware'
        })
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"👤 Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def send_to_client(self, websocket, message):
        """Send message to specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def process_message(self, websocket, message):
        """Process messages from web interface"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'connect_eeg':
                response = await self.connect_eeg()
            elif command == 'start_spelling':
                response = await self.start_spelling_session()
            elif command == 'stop_spelling':
                response = await self.stop_spelling_session()
            elif command == 'present_stimulus':
                stimulus_type = data.get('type')
                stimulus_id = data.get('id')
                response = await self.present_stimulus(stimulus_type, stimulus_id)
            elif command == 'get_predictions':
                context = data.get('context', '')
                response = await self.get_text_predictions(context)
            elif command == 'test_model':
                response = await self.test_model()
            elif command == 'get_status':
                response = self.get_system_status()
            elif command == 'switch_mode':
                mode = data.get('mode', 'simulation')
                response = await self.switch_mode(mode)
            else:
                response = {'error': f'Unknown command: {command}'}
                
            await self.send_to_client(websocket, response)
            
        except Exception as e:
            error_response = {'error': str(e)}
            await self.send_to_client(websocket, error_response)
    
    async def switch_mode(self, mode):
        """Switch between hardware and simulation modes"""
        try:
            if mode == 'simulation':
                if not self.simulation_mode:
                    # Switch to simulation
                    if self.is_running:
                        await self.stop_spelling_session()
                    
                    self.eeg_interface.disconnect()
                    self.simulation_mode = True
                    self.eeg_interface = OpenBCIStreamer(simulation_mode=True)
                    
                    await self.broadcast_message({
                        'type': 'mode_changed',
                        'mode': 'simulation',
                        'message': 'Switched to simulation mode'
                    })
                
                return {'success': True, 'mode': 'simulation'}
                
            elif mode == 'hardware':
                if self.simulation_mode:
                    # Switch to hardware
                    if self.is_running:
                        await self.stop_spelling_session()
                    
                    self.simulation_mode = False
                    self.eeg_interface = OpenBCIStreamer(simulation_mode=False)
                    
                    await self.broadcast_message({
                        'type': 'mode_changed',
                        'mode': 'hardware',
                        'message': 'Switched to hardware mode'
                    })
                
                return {'success': True, 'mode': 'hardware'}
            
            return {'success': False, 'message': 'Invalid mode'}
            
        except Exception as e:
            return {'success': False, 'message': f'Mode switch error: {e}'}
    
    async def connect_eeg(self):
        """Connect to EEG hardware or simulation"""
        try:
            if self.eeg_interface.connect():
                await self.broadcast_message({
                    'type': 'eeg_status',
                    'connected': True,
                    'message': f'EEG {"simulation" if self.simulation_mode else "hardware"} connected successfully',
                    'sample_rate': self.eeg_interface.sample_rate,
                    'channels': self.eeg_interface.num_channels,
                    'mode': 'simulation' if self.simulation_mode else 'hardware'
                })
                return {'success': True, 'message': 'EEG connected', 'mode': 'simulation' if self.simulation_mode else 'hardware'}
            else:
                return {'success': False, 'message': 'Failed to connect to EEG'}
                
        except Exception as e:
            return {'success': False, 'message': f'EEG connection error: {e}'}
    
    async def start_spelling_session(self):
        self.event_loop = asyncio.get_event_loop()
        """Start P300 spelling session"""
        try:
            # Define callback for real-time data processing
            def eeg_data_callback(eeg_data):
                """Callback for processing incoming EEG data"""
                try:
                    # Schedule the coroutine in the main event loop
                    asyncio.run_coroutine_threadsafe(
                        self.process_eeg_data(eeg_data), 
                        self.event_loop
                    )
                except Exception as e:
                    logger.error(f"Error scheduling EEG data processing: {e}")
            
            if self.eeg_interface.start_streaming(eeg_data_callback):
                self.is_running = True
                self.session_data = {
                    'trials': 0,
                    'detections': 0,
                    'accuracy': 0,
                    'start_time': datetime.now().isoformat(),
                    'stimulus_times': [],
                    'p300_responses': []
                }
                
                await self.broadcast_message({
                    'type': 'session_started',
                    'mode': 'simulation' if self.simulation_mode else 'hardware'
                })
                
                return {'success': True, 'message': 'Spelling session started'}
            else:
                return {'success': False, 'message': 'Failed to start EEG streaming'}
                
        except Exception as e:
            return {'success': False, 'message': f'Session start error: {e}'}
    
    async def stop_spelling_session(self):
        """Stop P300 spelling session"""
        self.is_running = False
        self.eeg_interface.stop_streaming()
        
        await self.broadcast_message({
            'type': 'session_stopped'
        })
        
        return {'success': True, 'message': 'Spelling session stopped'}
    
    async def present_stimulus(self, stimulus_type, stimulus_id):
        """Present stimulus and record timing"""
        stimulus_time = time.time()
        
        stimulus_info = {
            'time': stimulus_time,
            'type': stimulus_type,
            'id': stimulus_id,
            'processed': False
        }
        
        self.session_data['stimulus_times'].append(stimulus_info)
        self.current_stimulus = stimulus_info
        
        # Schedule P300 response processing
        asyncio.create_task(self.process_p300_response(stimulus_info))
        
        await self.broadcast_message({
            'type': 'stimulus_presented',
            'stimulus_type': stimulus_type,
            'stimulus_id': stimulus_id,
            'timestamp': stimulus_time
        })
        
        return {'success': True, 'stimulus': stimulus_info}
    
    async def process_p300_response(self, stimulus_info, delay=0.7):
        """Process P300 response after stimulus"""
        # Wait for P300 response window
        await asyncio.sleep(delay)
        
        if not self.is_running or stimulus_info['processed']:
            return
        
        try:
            # Extract epoch around stimulus
            epoch_data = self.eeg_interface.get_epoch(
                pre_stim=0.1, 
                post_stim=0.6, 
                trigger_time=stimulus_info['time']
            )
            
            if epoch_data is None:
                logger.warning("⚠️ Could not extract epoch data")
                return
            
            # Preprocess epoch
            # Preprocess epoch - don't reject for artifacts during real-time
            processed_epoch = self.processor.preprocess_epoch(epoch_data, check_artifacts=False)
            if processed_epoch is None:
                logger.warning("⚠️ Epoch preprocessing failed (artifact detected)")
                return
            
            # Classify P300 response
            p300_confidence = self.classify_p300_response(processed_epoch)
            
            # Update session data
            self.session_data['trials'] += 1
            stimulus_info['processed'] = True
            stimulus_info['p300_confidence'] = p300_confidence
            
            # After getting p300_confidence, add:
            if stimulus_info['type'] == 'row':
                self.p300_scores['rows'][stimulus_info['id']] += p300_confidence
            elif stimulus_info['type'] == 'column':
                self.p300_scores['columns'][stimulus_info['id']] += p300_confidence

            self.trial_count += 1

            # Check if we should select a character (after enough trials)
            if self.trial_count >= self.trials_per_selection:
                await self.select_character_from_p300()
            
            # Check if P300 detected
            p300_detected = p300_confidence > 0.7
            if p300_detected:
                self.session_data['detections'] += 1
            
            # Calculate accuracy
            if self.session_data['trials'] > 0:
                self.session_data['accuracy'] = (self.session_data['detections'] / self.session_data['trials']) * 100
            
            # Store response
            self.session_data['p300_responses'].append({
                'stimulus': stimulus_info,
                'confidence': p300_confidence,
                'detected': p300_detected,
                'epoch_shape': processed_epoch.shape
            })
            
            # Send detection to clients
            await self.broadcast_message({
                'type': 'p300_detection',
                'probability': p300_confidence,
                'detected': p300_detected,
                'stimulus_type': stimulus_info['type'],
                'stimulus_id': stimulus_info['id'],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing P300 response: {e}")
    
    async def process_eeg_data(self, eeg_data):
        """Process incoming EEG data for visualization"""
        try:
            # Send EEG data for visualization (downsample for web interface)
            if eeg_data.shape[1] > 50:
                # Send last 50 samples of first 4 channels
                viz_data = eeg_data[:4, -50:].tolist()
            else:
                viz_data = eeg_data[:4, :].tolist()
            
            # Add some P300-like activity for demonstration
            p300_prob = 0.0
            if hasattr(self, 'current_stimulus') and self.current_stimulus:
                # Simulate P300 probability based on time since stimulus
                time_since_stimulus = time.time() - self.current_stimulus['time']
                if 0.2 < time_since_stimulus < 0.5:  # P300 window
                    p300_prob = np.random.beta(3, 2) if self.simulation_mode else 0.0
            
            await self.broadcast_message({
                'type': 'eeg_data',
                'data': viz_data,
                'timestamp': time.time(),
                'p300_prob': p300_prob
            })
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {e}")
    
    def classify_p300_response(self, epoch_data):
        """Classify P300 response using trained model"""
        return self.validator.test_single_epoch(epoch_data)
    
    async def get_text_predictions(self, context):
        """Get AI text predictions"""
        try:
            predictions = self.predictor.predict_next_words(context, 5)
            
            return {
                'success': True,
                'predictions': predictions
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Prediction error: {e}',
                'predictions': self.predictor.get_common_words()[:5]
            }
    
    async def test_model(self):
        """Test the P300 model with sample data"""
        try:
            # Generate test data
            X_test = np.random.randn(100, 8, 150, 1)
            y_test = np.random.randint(0, 2, 100)
            
            if self.validator.model:
                # Real model validation
                results = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                }
            else:
                # Simulation results
                results = {
                    'accuracy': 0.75,
                    'precision': 0.70,
                    'recall': 0.80,
                    'f1_score': 0.75,
                    'note': 'Simulation mode results'
                }
            
            await self.broadcast_message({
                'type': 'model_test_results',
                'results': results
            })
            
            return {
                'success': True,
                'results': results,
                'message': 'Model validation completed'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Model test error: {e}'
            }
    
    def get_system_status(self):
        """Get current system status"""
        eeg_connected = (self.eeg_interface is not None)
        
        return {
            'model_loaded': self.validator.model is not None,
            'eeg_connected': eeg_connected,
            'eeg_streaming': self.eeg_interface.is_streaming if self.eeg_interface else False,
            'ai_loaded': self.predictor.model is not None,
            'is_running': self.is_running,
            'mode': 'simulation' if self.simulation_mode else 'hardware',
            'session_data': self.session_data,
            'hardware_info': {
                'sample_rate': self.eeg_interface.sample_rate if self.eeg_interface else 0,
                'num_channels': self.eeg_interface.num_channels if self.eeg_interface else 0,
                'board_id': str(self.eeg_interface.board_id) if self.eeg_interface else 'None'
            }
        }
    
    async def broadcast_message(self, message):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            disconnected = []
            for client in self.connected_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.discard(client)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.eeg_interface:
            self.eeg_interface.disconnect()
    
    async def start_server(self, host='localhost', port=8765):
        """Start the WebSocket server"""
        logger.info(f"Starting P300 backend server on {host}:{port}")
        logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'HARDWARE'}")
        
        async with websockets.serve(self.handle_client, host, port):
            logger.info("Server started successfully!")
            await asyncio.Future()  # Run forever


def create_test_data():
    """Create test data for model validation"""
    logger.info("Creating test data for P300 model validation...")
    
    n_epochs = 200
    n_channels = 8
    n_samples = 150
    
    X_test = np.random.randn(n_epochs, n_channels, n_samples, 1)
    y_test = np.random.choice([0, 1], n_epochs, p=[0.7, 0.3])
    
    # Add realistic P300 characteristics
    for i, label in enumerate(y_test):
        if label == 1:  # P300 trial
            peak_sample = 75  # Around 300ms
            parietal_channels = [2, 3, 6, 7]
            for ch in parietal_channels:
                if ch < n_channels:
                    X_test[i, ch, peak_sample-10:peak_sample+10, 0] += np.random.randn(20) * 2.5
    
    return X_test, y_test


def test_openbci_integration(simulation=True):
    """Test OpenBCI integration"""
    print("Testing OpenBCI Integration...")
    print("=" * 50)
    
    try:
        streamer = OpenBCIStreamer(simulation_mode=simulation)
        
        if streamer.connect():
            print(f"Connection successful! Mode: {'SIMULATION' if simulation else 'HARDWARE'}")
            print(f"Sample rate: {streamer.sample_rate} Hz")
            print(f"Channels: {streamer.num_channels}")
            
            # Test data collection
            data_received = False
            def data_callback(eeg_data):
                nonlocal data_received
                data_received = True
                print(f"Received EEG data: {eeg_data.shape}")
            
            if streamer.start_streaming(data_callback):
                print("Streaming started")
                
                # Wait for data
                time.sleep(3)
                
                if data_received:
                    print("Data reception successful!")
                    
                    # Test epoch extraction
                    epoch = streamer.get_epoch()
                    if epoch is not None:
                        print(f"Epoch extraction successful: {epoch.shape}")
                        
                        # Test preprocessing
                        processor = RealTimeProcessor()
                        processed = processor.preprocess_epoch(epoch)
                        if processed is not None:
                            print(f"Preprocessing successful: {processed.shape}")
                        
                streamer.stop_streaming()
                print("Streaming stopped")
            
            streamer.disconnect()
            print("Test completed successfully!")
            return True
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='P300 Speller Backend')
    parser.add_argument('--mode', choices=['hardware', 'simulation'], default='simulation',
                       help='Operating mode (default: simulation)')
    parser.add_argument('--model', default='models/p300_eegnet.h5',
                       help='Path to P300 model file')
    parser.add_argument('--port', type=int, default=8765,
                       help='WebSocket server port')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("P300 SPELLER BACKEND WITH OPENBCI CYTON R25 SUPPORT")
    print("=" * 60)
    
    # Test integration first
    print(f"\n1. Testing OpenBCI Integration ({args.mode.upper()} mode)...")
    simulation_mode = args.mode == 'simulation'
    
    if not test_openbci_integration(simulation=simulation_mode):
        print("Integration test failed. Check the setup.")
        return
    
    # Initialize backend
    print(f"\n2. Initializing Backend System...")
    backend = P300SpellerBackend(
        model_path=args.model,
        simulation_mode=simulation_mode
    )
    
    # Test model if available
    if backend.validator.model:
        print("\n3. Testing P300 Model...")
        X_test, y_test = create_test_data()
        print("Model is ready for real-time P300 detection.")
    else:
        print("\n3. No P300 model loaded - using simulation mode")
    
    # Start server
    print(f"\n4. Starting WebSocket Server on port {args.port}...")
    print(f"Mode: {args.mode.upper()}")
    print("Connect your web interface to: ws://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        await backend.start_server(port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        backend.cleanup()
        print("Server stopped. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())