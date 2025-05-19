import sys
import numpy as np
from numpy.fft import ifft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox,
                            QComboBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from ctypes import *
from dwfconstants import *
import time
import os
import collections

# Configure matplotlib for dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'

class AcquisitionThread(QThread):
    new_data_available = pyqtSignal(np.ndarray)
    
    def __init__(self, dwf_lib, hdwf, fs, chunk_size):
        super().__init__()
        self.dwf = dwf_lib
        self.hdwf = hdwf
        self.fs = fs
        self.chunk_size = chunk_size
        self.running = True
        
    def run(self):
        # Configure for continuous acquisition
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.fs))
        self.dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(self.chunk_size))
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
        
        while self.running:
            status = c_byte()
            while True:
                self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(status))
                if status.value == DwfStateDone.value:
                    break
                QThread.msleep(1)
            
            # Get available samples
            available = c_int()
            lost = c_int()
            corrupted = c_int()
            self.dwf.FDwfAnalogInStatusRecord(self.hdwf, byref(available), byref(lost), byref(corrupted))
            
            if available.value > 0:
                # Read new data
                samples = (c_double*available.value)()
                samples_ch2 = (c_double*available.value)()
                self.dwf.FDwfAnalogInStatusData(self.hdwf, 0, samples, available)
                self.dwf.FDwfAnalogInStatusData(self.hdwf, 1, samples_ch2, available)
                
                # Convert to numpy array and emit
                data_chunk = np.column_stack((np.array(samples), np.array(samples_ch2)))
                self.new_data_available.emit(data_chunk)
            
            QThread.msleep(10)

class RadarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Constants
        self.c = 3E8  # speed of light (m/s)
        self.Tp = 20E-3  # pulse duration (s)
        self.fstart = 2315E6  # LFM start frequency (Hz)
        self.fstop = 2536E6  # LFM stop frequency (Hz)
        self.BW = self.fstop - self.fstart  # bandwidth (Hz)
        self.Fs = 50000  # sampling rate (Hz)
        self.N = int(self.Tp * self.Fs)  # samples per pulse
        self.window = True  # Hamming window
        
        # Initialize DWF variables
        self.hdwf = c_int()
        self.sts = c_byte()
        self.hzAcq = c_double(self.Fs)
        self.nSamples = 0
        self.rgdSamples = None
        self.rgdSamples_ch2 = None
        self.recording = False
        self.data = None
        
        # Real-time processing variables
        self.data_buffer = collections.deque(maxlen=1000)
        self.processing_queue = collections.deque(maxlen=20)
        self.processing_busy = False
        self.realtime_mode = False
        self.acquisition_thread = None
        self.last_processed_time = 0
        self.processing_interval = 0.1  # seconds
        
        # Initialize UI
        self.init_ui()
        self.init_dwf()
        
    def init_ui(self):
        self.setWindowTitle("Radar Processor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Control panel
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        # Recording parameters
        self.rec_time_spin = QDoubleSpinBox()
        self.rec_time_spin.setRange(1, 300)
        self.rec_time_spin.setValue(25)
        self.rec_time_spin.setSuffix(" s")
        
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(1000, 100000)
        self.fs_spin.setValue(50000)
        self.fs_spin.setSuffix(" Hz")
        
        # Processing parameters
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setRange(1, 100)
        self.tp_spin.setValue(20)
        self.tp_spin.setSuffix(" ms")
        self.tp_spin.setSingleStep(1)
        
        self.window_check = QCheckBox("Apply Hamming Window")
        self.window_check.setChecked(True)
        
        # Trigger control
        self.trigger_spin = QDoubleSpinBox()
        self.trigger_spin.setRange(0.0, 5.0)
        self.trigger_spin.setValue(1.0)
        self.trigger_spin.setSuffix(" V")
        
        # Real-time controls
        self.processing_interval_spin = QDoubleSpinBox()
        self.processing_interval_spin.setRange(0.02, 1.0)
        self.processing_interval_spin.setValue(0.1)
        self.processing_interval_spin.setSuffix(" s")
        
        # Add colormap selection
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "jet", "plasma", "inferno", "magma", "turbo"])
        self.colormap_combo.setCurrentText("jet")  # Default to jet for compatibility
        self.colormap_combo.currentTextChanged.connect(self.process_data)
        
        # Add range compensation control
        self.compensation_spin = QDoubleSpinBox()
        self.compensation_spin.setRange(0.0, 1.0)
        self.compensation_spin.setValue(0.4)
        self.compensation_spin.setSingleStep(0.05)
        self.compensation_spin.valueChanged.connect(self.process_data)
        
        # Add dynamic range control
        self.dynamic_range_spin = QSpinBox()
        self.dynamic_range_spin.setRange(20, 60)
        self.dynamic_range_spin.setValue(40)
        self.dynamic_range_spin.setSingleStep(5)
        self.dynamic_range_spin.setSuffix(" dB")
        self.dynamic_range_spin.valueChanged.connect(self.process_data)
        
        # Buttons
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        self.realtime_btn = QPushButton("Start Real-Time Mode")
        self.realtime_btn.clicked.connect(self.toggle_realtime)
        
        self.process_btn = QPushButton("Process Data")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setEnabled(False)
        
        # Add Doppler processing button
        self.doppler_btn = QPushButton("Process Doppler")
        self.doppler_btn.clicked.connect(self.process_doppler)
        self.doppler_btn.setEnabled(False)
        
        self.save_btn = QPushButton("Save Data")
        self.save_btn.clicked.connect(self.save_data)
        self.save_btn.setEnabled(False)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        control_layout.addWidget(QLabel("Recording Time:"))
        control_layout.addWidget(self.rec_time_spin)
        control_layout.addWidget(QLabel("Sampling Frequency:"))
        control_layout.addWidget(self.fs_spin)
        control_layout.addWidget(QLabel("Pulse Duration:"))
        control_layout.addWidget(self.tp_spin)
        control_layout.addWidget(QLabel("Trigger Threshold:"))
        control_layout.addWidget(self.trigger_spin)
        control_layout.addWidget(self.window_check)
        control_layout.addWidget(QLabel("Processing Interval:"))
        control_layout.addWidget(self.processing_interval_spin)
        control_layout.addWidget(QLabel("Colormap:"))
        control_layout.addWidget(self.colormap_combo)
        control_layout.addWidget(QLabel("Range Compensation:"))
        control_layout.addWidget(self.compensation_spin)
        control_layout.addWidget(QLabel("Dynamic Range:"))
        control_layout.addWidget(self.dynamic_range_spin)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.realtime_btn)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.doppler_btn)  # Add Doppler button
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # Plot panel
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()
        self.rti_fig = Figure(figsize=(8, 6))
        self.rti_canvas = FigureCanvas(self.rti_fig)
        plot_layout.addWidget(self.rti_canvas)
        plot_panel.setLayout(plot_layout)
        
        # Add panels to main layout
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(plot_panel, 2)
    
    def init_dwf(self):
        if sys.platform.startswith("win"):
            self.dwf = cdll.dwf
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")
        
        version = create_string_buffer(16)
        self.dwf.FDwfGetVersion(version)
        print("DWF Version:", version.value.decode("ascii"))
        
        self.dwf.FDwfDeviceOpen(c_int(-1), byref(self.hdwf))
        
        if self.hdwf.value == 0:
            szerr = create_string_buffer(512)
            self.dwf.FDwfGetLastErrorMsg(szerr)
            self.status_label.setText(f"Failed to open device: {szerr.value.decode()}")
            self.record_btn.setEnabled(False)
            self.realtime_btn.setEnabled(False)
        else:
            self.status_label.setText("Device ready")
    
    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_btn.setText("Stop Recording")
            self.realtime_btn.setEnabled(False)
            self.process_btn.setEnabled(False)
            self.doppler_btn.setEnabled(False)  # Disable Doppler button when recording
            self.save_btn.setEnabled(False)
            self.status_label.setText("Recording...")
            
            # Start recording in a separate thread
            QTimer.singleShot(100, self.start_recording)
        else:
            # Stop recording
            self.recording = False
            self.record_btn.setText("Start Recording")
            self.realtime_btn.setEnabled(True)
            self.process_btn.setEnabled(True)
            self.doppler_btn.setEnabled(True)  # Enable Doppler button after recording
            self.save_btn.setEnabled(True)
            self.status_label.setText("Recording finished")
    
    def start_recording(self):
        # Set recording parameters
        tSampling = self.rec_time_spin.value()
        self.Fs = self.fs_spin.value()
        self.hzAcq = c_double(self.Fs)
        self.nSamples = int(self.Fs * tSampling)
        
        # Initialize sample arrays
        self.rgdSamples = (c_double*self.nSamples)()
        self.rgdSamples_ch2 = (c_double*self.nSamples)()
        
        # Set up acquisition
        self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(0), c_bool(True))
        self.dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(1), c_bool(True))
        self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(0), c_double(5))
        self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(1), c_double(5))
        self.dwf.FDwfAnalogInAcquisitionModeSet(self.hdwf, acqmodeRecord)
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, self.hzAcq)
        self.dwf.FDwfAnalogInRecordLengthSet(self.hdwf, c_double(tSampling))
        
        # Wait for offset to stabilize
        time.sleep(2)
        
        # Begin acquisition
        self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(0), c_int(1))
        
        # Record data
        cSamples = 0
        fLost = 0
        fCorrupted = 0
        
        while cSamples < self.nSamples and self.recording:
            self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(self.sts))
            
            if cSamples == 0 and (self.sts.value == DwfStateConfig or 
                                  self.sts.value == DwfStatePrefill or 
                                  self.sts.value == DwfStateArmed):
                continue
            
            cAvailable = c_int()
            cLost = c_int()
            cCorrupted = c_int()
            
            self.dwf.FDwfAnalogInStatusRecord(self.hdwf, byref(cAvailable), 
                                            byref(cLost), byref(cCorrupted))
            
            cSamples += cLost.value
            
            if cLost.value:
                fLost = 1
            if cCorrupted.value:
                fCorrupted = 1
            
            if cAvailable.value == 0:
                continue
            
            if cSamples + cAvailable.value > self.nSamples:
                break
            
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(0), 
                                           byref(self.rgdSamples, sizeof(c_double)*cSamples), 
                                           cAvailable)
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(1), 
                                           byref(self.rgdSamples_ch2, sizeof(c_double)*cSamples), 
                                           cAvailable)
            cSamples += cAvailable.value
            
            # Update status
            progress = min(100, int(cSamples / self.nSamples * 100))
            self.status_label.setText(f"Recording... {progress}%")
            QApplication.processEvents()
        
        if not self.recording:
            self.status_label.setText("Recording stopped")
            return
        
        # Create data array
        self.data = np.zeros((self.nSamples, 2))
        self.data[:, 0] = self.rgdSamples
        self.data[:, 1] = self.rgdSamples_ch2
        
        # Finish recording
        self.recording = False
        self.record_btn.setText("Start Recording")
        self.realtime_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.doppler_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        if fLost:
            self.status_label.setText("Recording finished (some samples lost)")
        elif fCorrupted:
            self.status_label.setText("Recording finished (some samples may be corrupted)")
        else:
            self.status_label.setText("Recording finished successfully")
    
    def toggle_realtime(self):
        if not self.realtime_mode:
            # Start real-time mode
            self.realtime_mode = True
            self.realtime_btn.setText("Stop Real-Time Mode")
            self.record_btn.setEnabled(False)
            self.doppler_btn.setEnabled(False)  # Disable Doppler button in real-time mode
            
            # Calculate optimal chunk size
            #min_chunk = int(self.Fs * 0.05)  # 50ms minimum
            pulse_chunk = int(self.Fs * self.Tp * 10.0)  # 5.0 pulse widths
            chunk_size = max(int(self.Fs * 0.1), pulse_chunk)  # at least 100ms worth of samples
          
            self.acquisition_thread = AcquisitionThread(self.dwf, self.hdwf, self.Fs, chunk_size)
            self.acquisition_thread.new_data_available.connect(self.handle_new_data)
            self.acquisition_thread.start()
            
            # Processing timer at 2x pulse rate
            self.processing_timer = QTimer()
            self.processing_timer.timeout.connect(self.safe_process_data)
            self.processing_timer.start(int(self.processing_interval_spin.value() * 1000))
            
            self.status_label.setText("Real-time mode active")
        else:
            # Stop real-time mode
            self.realtime_mode = False
            self.realtime_btn.setText("Start Real-Time Mode")
            self.record_btn.setEnabled(True)
            self.doppler_btn.setEnabled(True)  # Enable Doppler button when real-time mode stops
            
            if self.acquisition_thread:
                self.acquisition_thread.running = False
                self.acquisition_thread.wait()
                self.acquisition_thread = None
            
            if hasattr(self, 'processing_timer'):
                self.processing_timer.stop()
            
            self.status_label.setText("Real-time mode stopped")
    
    def handle_new_data(self, data_chunk):
        self.processing_queue.append(data_chunk)
    
    def safe_process_data(self):
        if self.processing_queue and not self.processing_busy:
            self.processing_busy = True
            try:
                # Combine all available chunks
                self.data = np.vstack(self.processing_queue)
                self.processing_queue.clear()
                
                # Process with current parameters
                self.process_data()
                
                # Update display
                QApplication.processEvents()
            except Exception as e:
                self.status_label.setText(f"Processing error: {str(e)}")
            finally:
                self.processing_busy = False
    
    def process_data(self):
        if self.data is None or len(self.data) == 0:
            return
        
        # Update processing parameters
        self.Tp = self.tp_spin.value() * 1e-3
        self.Fs = self.fs_spin.value()
        self.N = int(self.Tp * self.Fs)
        self.window = self.window_check.isChecked()
        
        s = self.data[:, 0]
        trig = self.data[:, 1]
        trig_threshold = self.trigger_spin.value()
        
        # Improved trigger detection
        trig[trig < trig_threshold] = 0
        trig[trig >= trig_threshold] = 1
        
        # Robust edge detection
        pulse_starts = []
        for j in range(10, len(trig) - self.N):
            if trig[j] == 1 and np.mean(trig[j-10:j]) < 0.1:  # Was completely low
                if j + self.N <= len(trig):
                    pulse_starts.append(j)
        
        pulse_count = len(pulse_starts)
        if pulse_count == 0:
            self.status_label.setText("No pulses detected! Adjust trigger level")
            return
        
        s2 = np.zeros([pulse_count, self.N])
        for i, start in enumerate(pulse_starts):
            s2[i, :] = s[start:start + self.N]
        
        # Pulse-to-pulse averaging
        for i in range(self.N):
            s2[:, i] -= np.mean(s2[:, i])
        
        # 2-pulse cancellation
        s3 = np.zeros_like(s2)
        for i in range(pulse_count - 1):
            s3[i, :] = s2[i + 1, :] - s2[i, :]
        
        pulse_count -= 1
        s3 = s3[0:pulse_count, :]
        
        # Apply window if selected
        if self.window:
            s3 *= np.hamming(self.N)
        
        # Range-Time-Intensity processing
        v = ifft(s3)
        
        # Apply range-dependent gain compensation
        # R^4 compensation factor (limited to avoid excessive noise amplification)
        range_bins = np.arange(1, self.N + 1)
        r4_compensation = np.minimum(np.power(range_bins/range_bins[0], 4), 100)
        r4_compensation = np.tile(r4_compensation, (pulse_count, 1))
        
        # Apply compensation (with user-adjustable factor)
        compensation_factor = self.compensation_spin.value()
        v = v * np.power(r4_compensation, compensation_factor)
        
        # Convert to dB scale
        v = 20 * np.log10(np.abs(v) + 1e-12)
        v = v[:, :self.N // 2]
        
        # Get dynamic range from control
        dynamic_range = self.dynamic_range_spin.value()
        
        # Normalize to max value
        m = np.max(v)
        grid = v - m
        
        # Calculate range axis in meters
        # Speed of light divided by twice the bandwidth gives range resolution
        range_res = self.c / (2 * self.BW)
        max_range = range_res * (self.N // 2)
        range_axis = np.linspace(0, max_range, self.N // 2)
        
        # Plot RTI with improved settings
        self.rti_fig.clear()
        ax = self.rti_fig.add_subplot(111)
        
        max_time = self.Tp * pulse_count
        
        # Set color range based on dynamic range control
        vmin = -dynamic_range
        vmax = 0
        
        # Get selected colormap
        cmap = self.colormap_combo.currentText()
        
        im = ax.imshow(grid, extent=[0, max_range, max_time, 0], 
                      aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        self.rti_fig.colorbar(im, ax=ax, label='dB')
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Time [s]')
        ax.set_title('RTI with 2-pulse clutter rejection and range compensation')
        
        # Adjust x-axis to focus on relevant range
        displayed_max_range = min(40, max_range)  # Show up to 40m or max range
        ax.set_xlim(0, displayed_max_range)
        
        # Add info text
        ax.text(0.02, 0.95, f"Pulses: {pulse_count}", transform=ax.transAxes,
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        
        self.rti_fig.tight_layout()
        self.rti_canvas.draw()
        
        self.status_label.setText(f"Processed {pulse_count} pulses with range compensation")
    
    def process_doppler(self):
        """
        Process velocity (Doppler) information using phase evolution between consecutive chirps.
        This implements the velocity measurement technique described in the FMCW radar document.
        """
        if self.data is None or len(self.data) == 0:
            return
        
        # Update processing parameters
        self.Tp = self.tp_spin.value() * 1e-3
        self.Fs = self.fs_spin.value()
        self.N = int(self.Tp * self.Fs)
        self.window = self.window_check.isChecked()
        
        s = self.data[:, 0]
        trig = self.data[:, 1]
        trig_threshold = self.trigger_spin.value()
        
        # Trigger detection
        trig[trig < trig_threshold] = 0
        trig[trig >= trig_threshold] = 1
        
        # Find pulse starts
        pulse_starts = []
        for j in range(10, len(trig) - self.N):
            if trig[j] == 1 and np.mean(trig[j-10:j]) < 0.1:
                if j + self.N <= len(trig):
                    pulse_starts.append(j)
        
        pulse_count = len(pulse_starts)
        if pulse_count == 0:
            self.status_label.setText("No pulses detected! Adjust trigger level")
            return
        
        # Extract pulse data
        s2 = np.zeros([pulse_count, self.N])
        for i, start in enumerate(pulse_starts):
            s2[i, :] = s[start:start + self.N]
        
        # Remove DC bias
        for i in range(self.N):
            s2[:, i] -= np.mean(s2[:, i])
        
        # Apply window if selected
        if self.window:
            window_func = np.hamming(self.N)
            for i in range(pulse_count):
                s2[i, :] *= window_func
        
        # Range processing - FFT along fast time
        range_fft = np.fft.fft(s2, axis=1)
        
        # Only use first half of FFT output (real signal)
        range_data = range_fft[:, :self.N//2]
        
        # Calculate range axis
        range_res = self.c / (2 * self.BW)
        max_range = range_res * (self.N // 2)
        range_axis = np.linspace(0, max_range, self.N // 2)
        
        # Need at least 2 pulses for velocity processing
        if pulse_count < 2:
            self.status_label.setText("Need at least 2 pulses for velocity processing")
            return
        
        # Velocity processing - FFT along slow time (pulse to pulse)
        # This captures the phase evolution between consecutive chirps
        # as described in the document section 11.1
        
        # Determine number of range bins to process for velocity
        # Limit to reasonable range where targets might be present
        max_useful_range_bin = min(self.N//2, int(40/range_res))
        
        # Prepare Range-Doppler map
        doppler_fft_size = min(64, pulse_count)  # Use power of 2 for efficiency
        range_doppler_map = np.zeros((max_useful_range_bin, doppler_fft_size), dtype=complex)
        
        # For each range bin, perform Doppler FFT across pulses
        for r_bin in range(max_useful_range_bin):
            # Extract phase evolution for this range bin across all pulses
            range_bin_data = range_data[:pulse_count, r_bin]
            
            # Apply window for Doppler processing
            if self.window:
                range_bin_data *= np.hamming(len(range_bin_data))
            
            # Zero-padding for better frequency resolution
            doppler_input = np.zeros(doppler_fft_size, dtype=complex)
            doppler_input[:pulse_count] = range_bin_data
            
            # FFT to extract Doppler frequencies
            doppler_output = np.fft.fft(doppler_input)
            
            # Store in Range-Doppler map
            range_doppler_map[r_bin, :] = doppler_output
        
        # Calculate velocity axis
        # Using equation v = (λ/4π) * (Δϕ/TC) with Doppler FFT
        lambda_m = self.c / ((self.fstart + self.fstop) / 2)  # Wavelength at center frequency
        
        # Velocity resolution is λ/(2*N*TC) according to equation 26
        v_res = lambda_m / (2 * pulse_count * self.Tp)
        
        # Maximum velocity is λ/(4*TC) according to equation 27
        v_max = lambda_m / (4 * self.Tp)
        
        # Create velocity axis
        velocity_axis = np.linspace(-v_max, v_max, doppler_fft_size)
        
        # Convert to dB for visualization
        range_doppler_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-12)
        
        # Normalize
        max_db = np.max(range_doppler_db)
        range_doppler_db = range_doppler_db - max_db
        
        # Get dynamic range from control
        dynamic_range = self.dynamic_range_spin.value()
        
        # Plot Range-Doppler map
        self.rti_fig.clear()
        ax = self.rti_fig.add_subplot(111)
        
        # Get selected color
# Get selected colormap
        cmap = self.colormap_combo.currentText()
        
        # Plot the Range-Doppler map
        im = ax.imshow(range_doppler_db, 
                      aspect='auto', 
                      cmap=cmap, 
                      vmin=-dynamic_range, 
                      vmax=0,
                      extent=[velocity_axis[0], velocity_axis[-1], max_useful_range_bin*range_res, 0])
        
        self.rti_fig.colorbar(im, ax=ax, label='dB')
        ax.set_xlabel('Velocity [m/s]')
        ax.set_ylabel('Range [m]')
        ax.set_title('Range-Doppler Map')
        
        # Add info text
        ax.text(0.02, 0.95, f"Pulses: {pulse_count}\nVel. res: {v_res:.2f} m/s", 
                transform=ax.transAxes,
                color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
        
        self.rti_fig.tight_layout()
        self.rti_canvas.draw()
        
        self.status_label.setText(f"Processed Range-Doppler map with {pulse_count} pulses")
    
    def save_data(self):
        if self.data is None:
            return
        
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Data File",
                                                  "",
                                                  "Data Files (*.dat);;All Files (*)",
                                                  options=options)
        if file_name:
            np.savetxt(file_name, self.data, delimiter=',')
            self.status_label.setText(f"Data saved to {os.path.basename(file_name)}")
    
    def closeEvent(self, event):
        if hasattr(self, 'hdwf') and self.hdwf.value != 0:
            self.dwf.FDwfDeviceClose(self.hdwf)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RadarGUI()
    window.show()
    sys.exit(app.exec_())