import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
from datetime import datetime
import os
import warnings
import librosa
warnings.filterwarnings('ignore')

baseline = 'machine'
anomaly = 'anomaly'
misc = 'non-machine'

class MachineSoundClassifier:
    """
    Enhanced Machine sound classification with UMA-16 v2 microphone array support.
    """
   
    def __init__(self, sample_rate=48000, duration=5, model_type='random_forest',
                 device_id=None, n_mfcc=20, use_uma16=True):
        """
        Initialize classifier with UMA-16 v2 support.
       
        Parameters:
        -----------
        sample_rate : int
            Sampling rate (UMA-16 v2 supports up to 48000 Hz)
        duration : int
            Recording duration in seconds
        model_type : str
            Type of classifier to use
        device_id : int
            Audio device ID (for UMA-16 ASIO driver)
        n_mfcc : int
            Number of MFCC coefficients
        use_uma16 : bool
            Enable UMA-16 specific processing (beamforming, etc.)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.model_type = model_type
        self.classifier = None
        self.feature_scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
        self.device_id = device_id
        self.device_info = None
        self.n_mfcc = n_mfcc
        self.use_uma16 = use_uma16
        self.n_channels = 16 if use_uma16 else 1  # UMA-16 has 16 channels
       
        self.augments = 4
        self.noise_factor = 0.0000000001
        self.pitch_floor = -1
        self.pitch_ceil = 1
        self.stretch_range = 0.05

        self.baseline_label = baseline
        self.misc_label = misc
        self.anomaly_label = anomaly

        self.data_set_path = 'data_bot_thread'
        
        # Get available devices
        self._list_devices()
        self._setup_device()
       
    def _list_devices(self):
        """List available audio devices including ASIO devices"""
        print("\n=== Available Audio Input Devices ===")
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    # Highlight UMA-16 device
                    if 'UMA' in device['name'] or 'miniDSP' in device['name'] or 'ASIO' in device['name']:
                        print(f"  🎤 Device {i}: {device['name']} (channels: {device['max_input_channels']}) - UMA-16 DETECTED")
                    else:
                        print(f"  Device {i}: {device['name']} (channels: {device['max_input_channels']})")
           
            # Check for ASIO specifically
            try:
                import sounddevice as sd_asio
                print("\n=== ASIO Devices (Recommended for UMA-16) ===")
                print("To use ASIO, install: pip install sounddevice[asio]")
            except:
                pass
               
        except Exception as e:
            print(f"  Error querying devices: {e}")
   
    def _setup_device(self):
        """Setup audio device with UMA-16 specific configuration"""
        try:
            # Try to auto-detect UMA-16 device
            if self.device_id is None:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] >= 16:
                        if 'UMA' in device['name'] or 'miniDSP' in device['name']:
                            self.device_id = i
                            self.device_info = device
                            self.n_channels = min(16, device['max_input_channels'])
                            print(f"\n✅ UMA-16 v2 detected!")
                            print(f"Using device {i}: {device['name']}")
                            print(f"Channels: {self.n_channels}")
                            break
           
            if self.device_id is not None:
                device_info = sd.query_devices(self.device_id, 'input')
                self.device_info = device_info
                print(f"\nUsing device: {device_info['name']}")
                print(f"Input channels: {device_info['max_input_channels']}")
                print(f"Default sample rate: {device_info['default_samplerate']}")
               
                # Use device's default sample rate if available
                if device_info['default_samplerate'] > 0:
                    self.sample_rate = int(device_info['default_samplerate'])
                    print(f"Setting sample rate to: {self.sample_rate} Hz")
            else:
                # Use default device
                self.device_id = sd.default.device[0]
                if self.device_id is None:
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            self.device_id = i
                            self.device_info = device
                            self.n_channels = min(self.n_channels, device['max_input_channels'])
                            print(f"\nUsing device: {device['name']}")
                            break
                else:
                    self.device_info = sd.query_devices(self.device_id, 'input')
                    print(f"\nUsing default device: {self.device_info['name']}")
                   
        except Exception as e:
            print(f"Error setting up device: {e}")
            print("Will try to auto-detect during recording...")
   
    def record_audio(self, label=""):
        """
        Record audio with UMA-16 v2 support.
        Returns single-channel audio after beamforming if UMA-16 is used.
        """
        try:
            print(f"\n🎤 Recording {label} for {self.duration} seconds...")
            # Standard recording
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.n_channels,
                dtype='float32',
                device=self.device_id
            )
            sd.wait()
            print("Recording complete!")

            data = []

            for i in range(self.n_channels):
                data.extend(recording[:,i])
            data = np.array(data)

            return data
           
        except sd.PortAudioError as e:
            print(f"PortAudio Error: {e}")
            print("\nTroubleshooting tips for UMA-16 v2:")
            print("1. Install ASIO support: pip install sounddevice[asio]")
            print("2. Select the miniDSP ASIO driver")
            print("3. Check Windows sound settings")
            print("4. Try different sample rate (48000 Hz recommended)")
           
            # Fallback: Create silent audio for testing
            print("\nCreating silent audio for testing...")
            return np.zeros(int(self.duration * self.sample_rate))
           
        except Exception as e:
            print(f"Unexpected error: {e}")
            return np.zeros(int(self.duration * self.sample_rate))
   
    def test_microphone(self):
        """Test UMA-16 microphone array and audio settings"""
        print("\n=== Testing UMA-16 v2 Microphone Array ===")
       
        if self.use_uma16:
            print(f"Testing {self.n_channels}-channel array...")
           
        print("Recording 2 seconds for testing...")
       
        try:
            if self.use_uma16 and self.n_channels == 16:
                # Test multi-channel recording
                test_recording = sd.rec(
                    int(2 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.n_channels,
                    dtype='float32',
                    device=self.device_id
                )
                sd.wait()

                # Check each channel
                print(f"\nChannel analysis:")
                active_channels = 0
                for i in range(self.n_channels):
                    max_amp = np.max(np.abs(test_recording[:, i]))
                    if max_amp > 0.01:
                        active_channels += 1
                        print(f"  Channel {i:2d}: max amplitude = {max_amp:.4f} ✓")
                    else:
                        print(f"  Channel {i:2d}: max amplitude = {max_amp:.4f} (low)")
               
                print(f"\nActive channels: {active_channels}/{self.n_channels}")
               
                if active_channels < 8:
                    print("⚠️ WARNING: Less than 8 channels active. Check UMA-16 connections.")
                else:
                    print("✅ UMA-16 v2 working correctly!")
               
            else:
                # Standard mono test
                test_recording = sd.rec(
                    int(2 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=self.device_id
                )
                sd.wait()
               
                max_amplitude = np.max(np.abs(test_recording))
                print(f"Max amplitude: {max_amplitude:.4f}")
               
                if max_amplitude < 0.01:
                    print("⚠️ WARNING: Very low audio input. Check microphone volume.")
                else:
                    print("✅ Microphone working correctly!")
           
            return True
           
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False
   
    def save_recording(self, audio, filename, label, path):
        """Save recording to file"""
        directory = os.path.join(self.data_set_path, path, label)
        if os.path.exists(directory):
            try:
                sf.write(os.path.join(directory, filename), audio, self.sample_rate)
                print(f"Saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print(f"Error saving file: Directory does not exist")
   
    def load_recording(self, filename):
        """Load recording from file"""
        try:
            audio, sr = sf.read(filename)
           
            # Handle multi-channel files
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                print(f"Loaded {audio.shape[1]}-channel audio")
                if self.use_uma16:
                    # Apply beamforming to multi-channel audio
                    audio, best_angle, _ = self.apply_beamforming_sweep(audio)
                    print(f"Applied beamforming, best angle: {best_angle}°")
                else:
                    # Convert to mono by averaging
                    audio = np.mean(audio, axis=1)
           
            # Resample if necessary
            if sr != self.sample_rate:
                print(f"Resampling from {sr} to {self.sample_rate}")
                audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
           
            # Adjust length
            target_length = int(self.sample_rate * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                audio = audio[:target_length]
               
            return audio
           
        except Exception as e:
            print(f"Error loading file: {e}")
            return np.zeros(int(self.sample_rate * self.duration))
   
    def load_recordings_from_folder(self, base_folder):
        """Load recordings from organized folder structure"""
        X_data = []
        y_data = []
       
        class_folders = {
            'baseline': 0,
            'anomaly': 1,
            'misc': 2
        }
       
        print(f"\n=== Loading recordings from {base_folder} ===")
       
        for class_name, label in class_folders.items():
            folder_path = os.path.join(base_folder, class_name)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path)
                        if f.endswith(('.wav', '.flac', '.mp3'))]
                print(f"Loading {len(files)} files from {class_name}...")
               
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    try:
                        audio = self.load_recording(file_path)
                        features = self.extract_features(audio)
                        X_data.append(features)
                        y_data.append(label)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
       
        if len(X_data) == 0:
            print("No recordings found!")
            return None, None
       
        X = np.array(X_data)
        y = np.array(y_data)
       
        print(f"\n✅ Loaded {len(X)} total recordings")
        print(f"  {self.baseline_label}: {np.sum(y == 0)}")
        print(f"  {self.anomaly_label}: {np.sum(y == 1)}")
        print(f"  {self.misc_label}: {np.sum(y == 2)}")
       
        return X, y
   
    def extract_features(self, audio):
        """Extract comprehensive features including MFCCs"""
        features_list = []
        feature_names_list = []
       
        # 1. Time-domain features
        energy = np.mean(audio ** 2)
        rms = np.sqrt(energy)
        peak = np.max(np.abs(audio))
        zero_crossing_rate = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
       
        features_list.extend([energy, rms, peak, zero_crossing_rate])
        feature_names_list.extend(['energy', 'rms', 'peak', 'zero_crossing_rate'])
       
        # 2. Statistical features
        mean_abs = np.mean(np.abs(audio))
        std_dev = np.std(audio)
        variance = np.var(audio)
       
        if np.std(audio) > 0:
            skewness = np.mean((audio - np.mean(audio)) ** 3) / (np.std(audio) ** 3)
            kurtosis = np.mean((audio - np.mean(audio)) ** 4) / (np.std(audio) ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
       
        features_list.extend([mean_abs, std_dev, variance, skewness, kurtosis])
        feature_names_list.extend(['mean_abs', 'std_dev', 'variance', 'skewness', 'kurtosis'])
       
        # 3. Frequency-domain features using librosa
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0].mean()
            zcr_librosa = librosa.feature.zero_crossing_rate(audio)[0].mean()
           
            features_list.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr_librosa])
            feature_names_list.extend(['spectral_centroid', 'spectral_rolloff',
                                      'spectral_bandwidth', 'zcr_librosa'])
        except:
            features_list.extend([0, 0, 0, 0])
            feature_names_list.extend(['spectral_centroid', 'spectral_rolloff',
                                      'spectral_bandwidth', 'zcr_librosa'])
       
        # 4. MFCC Features
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_means = np.mean(mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)
           
            features_list.extend(mfcc_means)
            features_list.extend(mfcc_stds)
           
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_mean_{i}')
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_std_{i}')
               
        except Exception as e:
            print(f"Error extracting MFCCs: {e}")
            features_list.extend([0] * (self.n_mfcc * 2))
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_mean_{i}')
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_std_{i}')
       
        # 5. Delta MFCCs
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
           
            delta_means = np.mean(mfcc_delta, axis=1)
            delta2_means = np.mean(mfcc_delta2, axis=1)
           
            features_list.extend(delta_means)
            features_list.extend(delta2_means)
           
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_delta_mean_{i}')
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_delta2_mean_{i}')
               
        except:
            features_list.extend([0] * (self.n_mfcc * 2))
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_delta_mean_{i}')
            for i in range(self.n_mfcc):
                feature_names_list.append(f'mfcc_delta2_mean_{i}')
       
        # Store feature names if not already stored
        if not self.feature_names:
            self.feature_names = feature_names_list
       
        return np.array(features_list)
   
    def compile_audio(self):
        """Compile training data from collected data"""

        aug_choice = input("Use data augmentation? (y/n, default=y): ").strip().lower()
        use_aug = aug_choice != 'n'

        directory = os.path.join(self.data_set_path, 'collected_data')

        labels = [l for l in os.listdir(directory)]

        for label in labels:

            files = [f for f in os.listdir(os.path.join(directory, label))]

            for file in files: 

                data, sr = sf.read(os.path.join(directory, label, file))

                filename = file.replace(".wav", "")
                filename = f"{filename}_0.wav"
                self.save_recording(data, filename, label, 'training_data')

                if use_aug:
                    for i in range(self.augments):
                        augmented = data

                        try:
                            pitch = np.random.randint(self.pitch_floor, 1 + self.pitch_ceil)
                            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch)
                        except:
                            pass

                        try:
                            stretch = np.random.uniform(1 - self.stretch_range, 1 + self.stretch_range)
                            augmented = librosa.effects.time_stretch(augmented, rate=stretch)
                        except:
                            pass

                        noise = np.random.randn(len(augmented))
                        augmented = augmented + self.noise_factor * noise

                        try:
                            filename = file.replace(".wav", "")
                            filename = f"{filename}_{i + 1}.wav"
                            self.save_recording(augmented, filename, label, 'training_data')
                        except:
                            pass
       
        return None
   
    def build_classifier(self):
        """Build sklearn classifier based on selected model_type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                subsample=0.8
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(2048, 1024, 512, 256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                batch_size=32
            )
        else:
            print(f"Unknown model type '{self.model_type}', using Random Forest")
            return RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
   
    def collect_training_data(self):
        """Interactive data collection for training"""
        print("\n=== Data Collection for Training ===")
        print("You will record sounds and label them as:")
        print(f"  1 - {self.baseline_label} sound (normal operation)")
        print(f"  2 - {self.misc_label} sound (background, speech, other machinery)")
        print(f"  3 - {self.anomaly_label} sound (faults, issues, deviations)")
        print("  q - Quit data collection\n")
       
        X_data = []
        y_data = []
        sample_count = 0
       
        while True:
            print(f"\n--- Sample {sample_count + 1} ---")
            choice = input(f"Label type (1={self.baseline_label}, 2={self.misc_label}, 3={self.anomaly_label}, q=quit): ").strip()
           
            if choice.lower() == 'q':
                break
           
            if choice not in ['1', '2', '3']:
                print("Invalid choice. Please enter 1, 2, 3, or q")
                continue
           
            # Record audio
            label_name = {'1': 'baseline', '2': 'misc', '3': 'anomaly'}[choice]
            audio = self.record_audio(label_name)

            print(audio)
           
            # Check if recording has sound
            if np.max(np.abs(audio)) < 0.01:
                print("⚠️ WARNING: Very low audio detected. Check microphone.")
                retry = input("Retry this recording? (y/n): ").strip().lower()
                if retry == 'y':
                    continue
           
            # Option to save recording
            save_choice = input(f"Save this recording? (y/n): ").strip().lower()
            if save_choice == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{label_name}_{timestamp}.wav"
                self.save_recording(audio, filename, label_name, 'collected_data')

                # Extract features
                features = self.extract_features(audio)
                X_data.append(features)
            
                # Label: 0 = machine, 1 = anomaly, 2 = non_machine
                label = 0 if choice == '1' else (1 if choice == '3' else 2)
                y_data.append(label)
            
                sample_count += 1
                print(f"✓ Sample {sample_count} collected and labeled as {label_name}")
            
            spec_choice = input(f"View spectogram of recordings? (y/n): ").strip().lower()
            if spec_choice == 'y':
                
                directory = os.path.join(self.data_set_path, 'collected_data', label_name)
                if os.path.exists(directory):
                    samplerate1, data1 = wf.read(os.path.join(directory, filename))

                eps = 1e-12

                nfft = 2048                 # FFT-storlek per tidsruta (ger frekvensupplösning)
                noverlap = int(nfft * 0.75) # överlapp mellan rutor (ger mjukare tid)
                window = np.hanning(nfft)   # "Gabor-ish" fönster
                
                plt.figure(figsize=(18, 4))

                Pxx, f, tt, im = plt.specgram(
                    data1,
                    NFFT=nfft,
                    Fs=samplerate1,
                    window=window,
                    noverlap=noverlap,
                    detrend='mean',
                    scale='dB',     # visar i dB
                    mode='psd'      # power spectral density
                )

                plt.title("Spektrogram (Original)")
                plt.xlabel("Tid (s)")
                plt.ylabel("Frekvens (Hz)")
                plt.ylim(20, samplerate1 // 2)

                # Gör färgskalan lite rimligare (valfritt men ofta bättre)
                # Du kan kommentera bort om du vill.
                vmin = np.percentile(10 * np.log10(Pxx + eps), 5)
                vmax = np.percentile(10 * np.log10(Pxx + eps), 99)
                im.set_clim(vmin, vmax)

                plt.colorbar(im, label="Nivå (dB)")

                plt.show()

            # Show summary
            total_samples = len(y_data)
            print(f"\nCollected so far (including augmented):")
            print(f"  {self.baseline_label} (0): {y_data.count(0)}")
            print(f"  {self.baseline_label} (1): {y_data.count(1)}")
            print(f"  {self.misc_label} (2): {y_data.count(2)}")
            print(f"  Total: {total_samples}")
       
        if len(X_data) == 0:
            print("No data collected. Exiting.")
            return None, None
       
        X = np.array(X_data)
        y = np.array(y_data)
       
        print(f"\n✅ Data collection complete!")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
       
        return X, y
   
    def train_classifier(self, X_train, y_train, test_size=0.2, perform_grid_search=False):
        """Train the classifier with optional grid search"""
        if X_train is None or y_train is None:
            print("No training data provided!")
            return None
       
        if len(X_train) < 10:
            print("⚠️ WARNING: Very few training samples. Collect at least 10 samples for better results.")
       
        # Split data
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=min(test_size, 0.3), random_state=42, stratify=y_train
        )
       
        # Normalize features
        X_train_scaled = self.feature_scaler.fit_transform(X_train_split)
        X_val_scaled = self.feature_scaler.transform(X_val_split)
       
        # Build classifier
        self.classifier = self.build_classifier()
       
        print(f"\n=== Training {self.model_type.upper()} Classifier ===")
        print(f"Training samples: {len(X_train_scaled)}")
        print(f"Validation samples: {len(X_val_scaled)}")
        print(f"Features: {X_train_scaled.shape[1]}")
       
        # Optional grid search
        if perform_grid_search and self.model_type == 'random_forest':
            print("\nPerforming grid search for hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(
                self.classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train_split)
            self.classifier = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.2%}")
        else:
            self.classifier.fit(X_train_scaled, y_train_split)
       
        # Validate
        y_pred = self.classifier.predict(X_val_scaled)
       
        print(f"\n✅ Training Complete!")
        print(f"Validation Accuracy: {accuracy_score(y_val_split, y_pred):.2%}")
        print("\nClassification Report:")
        print(classification_report(y_val_split, y_pred,
                                   target_names=[self.baseline_label, self.anomaly_label, self.misc_label]))
       
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val_split, y_pred)
        print(cm)
       
        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train_split, cv=min(5, len(y_train_split)))
        print(f"\n{len(cv_scores)}-fold Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
       
        return self.classifier
   
    def classify_sound(self, audio):
        """Classify a sound recording"""
        if self.classifier is None:
            raise ValueError("Classifier not trained. Train first with your data.")
       
        # Extract features
        features = self.extract_features(audio)
        features = features.reshape(1, -1)
       
        # Normalize
        features = self.feature_scaler.transform(features)
       
        # Predict class
        class_idx = self.classifier.predict(features)[0]
       
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = probabilities[class_idx]
            all_probs = {
                self.baseline_label: probabilities[0],
                self.anomaly_label: probabilities[1],
                self.misc_label: probabilities[2]
            }
        else:
            confidence = 1.0
            all_probs = {self.baseline_label: 0, self.anomaly_label: 0, self.misc_label: 0}
       
        class_labels = {0: self.baseline_label, 1: self.anomaly_label, 2: self.misc_label}
       
        return {
            'class': class_labels[class_idx],
            'class_index': class_idx,
            'confidence': confidence,
            'probabilities': all_probs,
            'is_machine': class_idx == 0,
            'is_anomaly': class_idx == 1,
            'is_non_machine': class_idx == 2
        }
   
    def save_model(self, filepath='machine_classifier'):
        """Save trained model and scaler"""
        if self.classifier:
            model_data = {
                'classifier': self.classifier,
                'scaler': self.feature_scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'n_mfcc': self.n_mfcc,
                'use_uma16': self.use_uma16,
                'n_channels': self.n_channels
            }
            with open(f'{filepath}.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved to {filepath}.pkl")
   
    def load_model(self, filepath='machine_classifier'):
        """Load trained model and scaler"""
        try:
            with open(f'{filepath}.pkl', 'rb') as f:
                model_data = pickle.load(f)
           
            self.classifier = model_data['classifier']
            self.feature_scaler = model_data['scaler']
            self.model_type = model_data.get('model_type', self.model_type)
            self.feature_names = model_data.get('feature_names', [])
            self.n_mfcc = model_data.get('n_mfcc', 20)
            self.use_uma16 = model_data.get('use_uma16', False)
            self.n_channels = model_data.get('n_channels', 1)
            print(f"✅ Model loaded from {filepath}.pkl")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath}.pkl not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
   
    def get_feature_importance(self, top_n=20):
        """Get feature importance for tree-based models"""
        if self.classifier is None:
            print("No trained model available")
            return None
       
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
           
            print(f"\n=== Top {top_n} Feature Importances ===")
            for i in range(min(top_n, len(indices))):
                idx = indices[i]
                if idx < len(self.feature_names):
                    print(f"{i+1:2d}. {self.feature_names[idx]:30s}: {importances[idx]:.4f}")
           
            return importances
        else:
            print(f"Model type '{self.model_type}' doesn't provide feature importances")
            return None
        
    def change_internals(self):
        """Change internal values while the program is still running"""
        while True:
            print("\n Choose variable")
            print("1. Noise factor")
            print("2. Minimum pitch")
            print("3. Maximum pitch")
            print("4. Time stretch range")
            print("5. Number of augment files")
            print("6. Back")

            value_choice = input("\nSelect option: ").strip()

            if value_choice == '1':
                print(f"\nCurrent value: {self.noise_factor}")
                self.noise_factor = float(input("Input new value: ").strip() or self.noise_factor)
                print(f"\nValue changed to: {self.noise_factor}")
            elif value_choice == '2':
                print(f"\nCurrent value: {self.pitch_floor}")
                self.pitch_floor = int(input("Input new value: ").strip() or self.pitch_floor)
                print(f"\nValue changed to: {self.pitch_floor}")
            elif value_choice == '3':
                print(f"\nCurrent value: {self.pitch_ceil}")
                self.pitch_ceil = int(input("Input new value: ").strip() or self.pitch_ceil)
                print(f"\nValue changed to: {self.pitch_ceil}")
            elif value_choice == '4':
                print(f"\nCurrent value: {self.stretch_range}")
                self.stretch_range = float(input("Input new value: ").strip() or self.stretch_range)
                print(f"\nValue changed to: {self.stretch_range}")
            elif value_choice == '5':
                print(f"\nCurrent value: {self.augments}")
                self.augments = int(input("Input new value: ").strip() or self.augments)
                print(f"\nValue changed to: {self.augments}")
            elif value_choice == '6':
                break
            else:
                print("Invalid choice")


# ============= MAIN FUNCTION =============

def main():
    """Main function with UMA-16 v2 support"""
    print("=== Enhanced Machine Sound Classification with UMA-16 v2 Support ===")
    print("\nThis system supports the UMA-16 v2 16-channel microphone array")
    
    use_uma16 = False

    # Try to list audio devices
    try:
        print("\nDetecting audio devices...")
        devices = sd.query_devices()
        input_devices = []
        uma16_devices = []
       
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                input_devices.append(i)
                if ('UMA' in d['name'] or 'miniDSP' in d['name']) and d['max_input_channels'] >= 16:
                    uma16_devices.append(i)
       
        if not input_devices:
            print("❌ No input devices found! Please check your microphone.")
            return
       
        print(f"Found {len(input_devices)} input device(s)")
        if uma16_devices:
            print(f"🎤 UMA-16 compatible devices: {uma16_devices}")
       
    except Exception as e:
        print(f"Error detecting devices: {e}")
        return
   
    # Let user select device
    device_id = None
    if len(input_devices) > 1:
        print("\nAvailable input devices:")
        for i in input_devices:
            is_uma = i in uma16_devices
            uma_tag = " 🎤 UMA-16" if is_uma else ""
            print(f"  Device {i}: {devices[i]['name']} (channels: {devices[i]['max_input_channels']}){uma_tag}")
       
        choice = input(f"\nSelect device (0-{max(input_devices)}) or press Enter for default: ").strip()
        if choice:
            device_id = int(choice)

    if devices[device_id]['index'] in uma16_devices:
        print("\nUMA-16 device selected")
        use_uma16 = True
   
    # Model selection
    print("\nSelect model type:")
    print("1. Neural Network (MLP)")
    print("2. Gradient Boosting")
    print("3. SVM")
    print("4. Random Forest")
    model_choice = input("Choice (1-4, default=1): ").strip()
   
    model_types = {
        '1': 'mlp',
        '2': 'gradient_boosting',
        '3': 'svm',
        '4': 'random_forest'
    }
    model_type = model_types.get(model_choice, 'mlp')
   
    # Sample rate selection
    sample_rate = 48000  # UMA-16 v2 optimal rate
    if not use_uma16:
        rate_choice = input("\nSample rate (22050 or 48000, default=22050): ").strip()
        sample_rate = int(rate_choice) if rate_choice in ['22050', '48000'] else 22050
   
    # Initialize classifier
    classifier = MachineSoundClassifier(
        sample_rate=sample_rate,
        duration=3,
        model_type=model_type,
        device_id=device_id,
        n_mfcc=20,
        use_uma16=use_uma16
    )
   
    # Main menu
    while True:
        print("\n" + "="*50)
        print("Main Menu:")
        print("1. Test microphone")
        print("2. Collect training data")
        print("3. Compile data")
        print("4. Train model")
        print("5. Classify a sound")
        print("6. Save model")
        print("7. Load model")
        print("8. Show feature importance")
        print("9. Batch classify from folder")
        print("10. Modify internal values")
        print("11. Change setup")
        print("12. Clear training data")
        print("x. Exit")
       
        choice = input("\nSelect option: ").strip()
       
        if choice == '1':
            classifier.test_microphone()
           
        elif choice == '2':
            X, y = classifier.collect_training_data()
            if X is not None:
                print(f"\nCollected {len(X)} samples. Ready to train!")
        
        elif choice == '3':
            classifier.compile_audio()
            print("\nAudio samples compiled and ready for training.")

        elif choice == '4':
            folder = input(f"Enter folder path with 'baseline', 'anomaly', 'misc' subfolders: ").strip()
            if os.path.exists(folder):
                X, y = classifier.load_recordings_from_folder(folder)
                if X is not None:
                    grid_choice = input("Perform grid search? (y/n, default=n): ").strip().lower()
                    perform_gs = grid_choice == 'y'
                    classifier.train_classifier(X, y, perform_grid_search=perform_gs)
            else:
                print(f"Folder not found: {folder}")
           
        elif choice == '5':
            if classifier.classifier is None:
                print("❌ No trained model. Please train first (option 4)")
            else:
                input("\nPress Enter to record sound for classification...")
                audio = classifier.record_audio("test")
                result = classifier.classify_sound(audio)
               
                print(f"\n{'='*40}")
                print(f"Classification: {result['class'].upper()}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"\nDetailed probabilities:")
                for cls, prob in result['probabilities'].items():
                    print(f"  {cls}: {prob:.1%}")
                print(f"{'='*40}")
               
                if result['is_machine']:
                    print("✓ Normal operation")
                elif result['is_anomaly']:
                    print("⚠️ ANOMALY detected!")
                else:
                    print("✗ Miscellaneous sound detected")
                   
        elif choice == '6':
            if classifier.classifier:
                name = input("Model name (default: my_model): ").strip()
                if not name:
                    name = "my_model"
                classifier.save_model(name)
            else:
                print("No model to save. Train first.")
               
        elif choice == '7':
            name = input("Model filename (without .pkl): ").strip()
            if not name:
                name = "my_model"
            classifier.load_model(name)
           
        elif choice == '8':
            classifier.get_feature_importance(top_n=20)
           
        elif choice == '9':
            if classifier.classifier is None:
                print("❌ No trained model. Please train first (option 3)")
            else:
                folder = input("Enter folder path with audio files to classify: ").strip()
                if os.path.exists(folder):
                    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.flac', '.mp3'))]
                    print(f"\nClassifying {len(files)} files...")
                   
                    results = {baseline: 0, anomaly: 0, misc: 0}
                    for file in files:
                        file_path = os.path.join(folder, file)
                        try:
                            audio = classifier.load_recording(file_path)
                            result = classifier.classify_sound(audio)
                            results[result['class']] += 1
                            print(f"{file}: {result['class']} ({result['confidence']:.1%})")
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                   
                    print(f"\nSummary:")
                    for cls, count in results.items():
                        print(f"  {cls}: {count}")
                else:
                    print(f"Folder not found: {folder}")
        
        elif choice == '10':
            classifier.change_internals()

        elif choice == '11':
            while True:
                print("\nClassifier setup settings")
                print("1. Change microphone")
                print("2. Change model")
                print("3. Back")

                choice = input("\nSelect option: ").strip()

                if choice == '1':
                    use_uma16 = False

                    # Try to list audio devices
                    try:
                        print("\nDetecting audio devices...")
                        devices = sd.query_devices()
                        input_devices = []
                        uma16_devices = []
                    
                        for i, d in enumerate(devices):
                            if d['max_input_channels'] > 0:
                                input_devices.append(i)
                                if ('UMA' in d['name'] or 'miniDSP' in d['name']) and d['max_input_channels'] >= 16:
                                    uma16_devices.append(i)
                    
                        if not input_devices:
                            print("❌ No input devices found! Please check your microphone.")
                            return
                    
                        print(f"Found {len(input_devices)} input device(s)")
                        if uma16_devices:
                            print(f"🎤 UMA-16 compatible devices: {uma16_devices}")
                    
                    except Exception as e:
                        print(f"Error detecting devices: {e}")
                        return
                
                    # Let user select device
                    device_id = None
                    if len(input_devices) > 1:
                        print("\nAvailable input devices:")
                        for i in input_devices:
                            is_uma = i in uma16_devices
                            uma_tag = " 🎤 UMA-16" if is_uma else ""
                            print(f"  Device {i}: {devices[i]['name']} (channels: {devices[i]['max_input_channels']}){uma_tag}")
                    
                        choice = input(f"\nSelect device (0-{max(input_devices)}) or press Enter for default: ").strip()
                        if choice:
                            device_id = int(choice)

                    if devices[device_id]['index'] in uma16_devices:
                        print("\nUMA-16 device selected")
                        use_uma16 = True
                    
                    # Sample rate selection
                    sample_rate = 48000  # UMA-16 v2 optimal rate
                    if not use_uma16:
                        rate_choice = input("\nSample rate (22050 or 48000, default=22050): ").strip()
                        sample_rate = int(rate_choice) if rate_choice in ['22050', '48000'] else 22050

                elif choice == '2':
                    # Model selection
                    print("\nSelect model type:")
                    print("1. Neural Network (MLP)")
                    print("2. Gradient Boosting")
                    print("3. SVM")
                    print("4. Random Forest")
                    model_choice = input("Choice (1-4, default=1): ").strip()
                
                    model_types = {
                        '1': 'mlp',
                        '2': 'gradient_boosting',
                        '3': 'svm',
                        '4': 'random_forest'
                    }
                    model_type = model_types.get(model_choice, 'mlp')

                elif choice == '3':
                    break

                else:
                    print("Invalid choice")
                    continue

                classifier = MachineSoundClassifier(
                    sample_rate=sample_rate,
                    duration=3,
                    model_type=model_type,
                    device_id=device_id,
                    n_mfcc=20,
                    use_uma16=use_uma16
                )

        elif choice == '12':
            print("\nContinuing will remove all sound files from the traning data folder.")

            choice = input("\nDo you want to continue (y/n)? ").strip()

            if choice == 'y':

                print("\nRemoving files...")
                directory = os.path.join('data_set', 'training_data')

                labels = [l for l in os.listdir(directory)]

                for label in labels:

                    files = [f for f in os.listdir(os.path.join(directory, label))]

                    for file in files:

                        filepath = os.path.join(directory, label, file)
                        os.remove(filepath)
                
                print("All files removed!")

        elif choice == 'x':
            print("Goodbye!")
            break
       
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
