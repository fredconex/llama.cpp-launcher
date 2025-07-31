#!/usr/bin/env python3
"""
Enhanced Llama.cpp Launcher with PyQt6 GUI
Features:
- Dark theme interface
- Model selection with GGUF file scanning
- Parameter customization
- Tensor visualization chart
- Settings persistence (global and per-model)
- Regex filtering for tensor override patterns
"""

import sys
import os
import json
import re
import subprocess
import glob
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QCheckBox, QComboBox, QPushButton, QFileDialog, QTextEdit,
    QTabWidget, QGroupBox, QScrollArea, QFrame, QSplitter,
    QListWidget, QListWidgetItem, QProgressBar, QSlider,
    QMessageBox, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPixmap, QIcon

# Configuration classes
@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_path: str = ""
    ctx_size: int = 4096
    n_gpu_layers: int = 0
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: int = -1
    flash_attn: bool = False
    tensor_override: str = ""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.0
    system_prompt: str = ""
    chat_template: str = ""
    # Server-specific parameters
    server_host: str = "127.0.0.1"
    server_port: int = 8080

@dataclass
class GlobalConfig:
    """Global application configuration"""
    models_directory: str = ""
    executable_path: str = ""
    last_model: str = ""
    window_geometry: str = ""
    theme: str = "dark"
    auto_detect_layers: bool = True

class TensorVisualizationWidget(QWidget):
    """Enhanced widget to visualize tensor allocation with detailed layer breakdown"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)  # Space for tensor visualization
        self.setMaximumHeight(350)  # Space for tensor visualization
        self.total_layers = 0
        self.gpu_layers = 0
        self.tensor_pattern = ""
        self.tensor_names = []
        self.layer_tensor_map = {}  # Maps layer index to list of tensor types
        self.pattern_matches = {}   # Cache for pattern matching results
        self.layer_blocks = {}      # Store block positions for mouse interaction
        
        # Enable mouse tracking for tooltips
        self.setMouseTracking(True)
        
        # Common tensor types per layer in transformer models
        self.common_tensor_types = [
            "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
            "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"
        ]
        
    def set_layers(self, total: int, gpu: int, pattern: str = "", tensor_names: List[str] = None):
        """Update layer visualization with detailed tensor information"""
        self.total_layers = total
        self.gpu_layers = gpu
        self.tensor_pattern = pattern
        self.tensor_names = tensor_names or []
        
        # Build layer-to-tensor mapping from actual tensor names
        self._build_layer_tensor_map()
        
        # Clear pattern match cache when pattern changes
        if pattern != getattr(self, '_last_pattern', ''):
            self.pattern_matches.clear()
            self._last_pattern = pattern
            
        self.update()
        
    def _build_layer_tensor_map(self):
        """Build mapping of layers to their tensor components"""
        self.layer_tensor_map.clear()
        
        # Initialize with common tensor types for each layer
        for i in range(self.total_layers):
            self.layer_tensor_map[i] = self.common_tensor_types.copy()
            
        # If we have actual tensor names, use them to refine the mapping
        if self.tensor_names:
            layer_tensors = {}
            for tensor_name in self.tensor_names:
                # Extract layer number and tensor type
                match = re.search(r'blk\.(\d+)\.(.+)', tensor_name)
                if match:
                    layer_idx = int(match.group(1))
                    tensor_type = match.group(2)
                    
                    if layer_idx not in layer_tensors:
                        layer_tensors[layer_idx] = set()
                    layer_tensors[layer_idx].add(tensor_type)
            
            # Update mapping with actual tensor types found
            for layer_idx, tensor_types in layer_tensors.items():
                if layer_idx < self.total_layers:
                    self.layer_tensor_map[layer_idx] = list(tensor_types)
        
    def paintEvent(self, event):
        """Paint event with tensor visualization and clear GPU/CPU indication"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.total_layers == 0:
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No model loaded")
            return
            
        # Calculate dimensions
        rect = self.rect()
        margin = 10
        legend_height = 45  # Space for legend
        available_width = rect.width() - 2 * margin
        available_height = rect.height() - margin - legend_height
        
        # Calculate layout - show layers with tensor details
        blocks_per_row = min(12, self.total_layers)
        rows = (self.total_layers + blocks_per_row - 1) // blocks_per_row
        block_width = available_width // blocks_per_row
        block_height = max(40, available_height // rows)  # Ensure minimum height
        
        # Ensure minimum block size for readability
        block_width = max(block_width, 60)
        
        # Adjust available height if blocks would be too tall
        max_blocks_height = rows * block_height
        if max_blocks_height > available_height:
            block_height = available_height // rows
        
        # Clear previous block positions
        self.layer_blocks.clear()
        
        # Draw layer blocks with tensor details
        for i in range(self.total_layers):
            row = i // blocks_per_row
            col = i % blocks_per_row
            
            x = margin + col * block_width
            y = margin + row * block_height
            
            # Store block position for mouse interaction
            self.layer_blocks[i] = (x, y, block_width - 2, block_height - 2)
            
            self._draw_layer_block(painter, x, y, block_width - 2, block_height - 2, i)
            
        # Draw enhanced legend and pattern info
        self._draw_legend(painter, rect)
        
    def _draw_layer_block(self, painter, x, y, width, height, layer_idx):
        """Draw a layer block with tensor breakdown and clear GPU/CPU indication"""
        # Get tensor types for this layer
        tensor_types = self.layer_tensor_map.get(layer_idx, self.common_tensor_types)
        
        # Calculate tensor subdivision
        tensors_per_row = min(3, len(tensor_types))
        tensor_rows = (len(tensor_types) + tensors_per_row - 1) // tensors_per_row
        tensor_width = width // tensors_per_row
        tensor_height = height // (tensor_rows + 1)  # +1 for layer label
        
        # Draw layer label background with clear GPU/CPU indication
        label_rect = (x, y, width, tensor_height)
        if layer_idx < self.gpu_layers:
            label_color = QColor(76, 175, 80)  # Solid green for GPU
            allocation_text = "GPU"
        else:
            label_color = QColor(244, 67, 54)  # Solid red for CPU
            allocation_text = "CPU"
            
        painter.fillRect(*label_rect, label_color)
        
        # Draw layer number and allocation type
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x + 2, y + tensor_height - 2, f"L{layer_idx}")
        
        # Draw allocation type on the right if space allows
        if width > 50:
            font.setBold(False)
            font.setPointSize(max(6, font.pointSize() - 1))
            painter.setFont(font)
            painter.drawText(x + width - 25, y + tensor_height - 2, allocation_text)
        
        # Draw individual tensors
        for i, tensor_type in enumerate(tensor_types[:9]):  # Limit to 9 tensors for display
            tensor_row = i // tensors_per_row
            tensor_col = i % tensors_per_row
            
            tensor_x = x + tensor_col * tensor_width
            tensor_y = y + tensor_height + tensor_row * tensor_height
            
            # Determine tensor color based on override pattern
            tensor_color = self._get_tensor_color(layer_idx, tensor_type)
            
            painter.fillRect(tensor_x, tensor_y, tensor_width - 1, tensor_height - 1, tensor_color)
            
            # Draw tensor type abbreviation if space allows
            if tensor_width > 20 and tensor_height > 15:
                painter.setPen(QColor(255, 255, 255))
                font.setPointSize(max(6, font.pointSize()))
                painter.setFont(font)
                abbrev = self._get_tensor_abbreviation(tensor_type)
                painter.drawText(tensor_x + 1, tensor_y + tensor_height - 2, abbrev)
                
    def _get_tensor_color(self, layer_idx, tensor_type):
        """Get color for a specific tensor based on allocation rules"""
        # Check cache first
        cache_key = (layer_idx, tensor_type, self.tensor_pattern)
        if cache_key in self.pattern_matches:
            return self.pattern_matches[cache_key]
        
        # Check if tensor override pattern affects this specific tensor
        if self.tensor_pattern and self._matches_tensor_pattern(layer_idx, tensor_type):
            # Pattern override - determine actual allocation type from pattern
            # Most override patterns force tensors to CPU, so use red
            override_color = QColor(244, 67, 54)  # Red for CPU (overridden tensors typically go to CPU)
            self.pattern_matches[cache_key] = override_color
            return override_color
        
        # Default allocation based on GPU layers
        if layer_idx < self.gpu_layers:
            base_color = QColor(76, 175, 80)  # Green for GPU
        else:
            base_color = QColor(244, 67, 54)  # Red for CPU
            
        self.pattern_matches[cache_key] = base_color
        return base_color
        
    def _get_tensor_abbreviation(self, tensor_type):
        """Get short abbreviation for tensor type"""
        abbreviations = {
            "attn_norm": "AN",
            "attn_q": "Q",
            "attn_k": "K", 
            "attn_v": "V",
            "attn_output": "AO",
            "ffn_norm": "FN",
            "ffn_gate": "FG",
            "ffn_up": "FU",
            "ffn_down": "FD",
            "weight": "W",
            "bias": "B"
        }
        
        # Try exact match first
        if tensor_type in abbreviations:
            return abbreviations[tensor_type]
            
        # Try partial matches
        for key, abbrev in abbreviations.items():
            if key in tensor_type:
                return abbrev
                
        # Fallback to first 2 characters
        return tensor_type[:2].upper()
        
    def _matches_tensor_pattern(self, layer_idx: int, tensor_type: str) -> bool:
        """Enhanced pattern matching for specific tensors"""
        if not self.tensor_pattern:
            return False
            
        try:
            # Build full tensor name for pattern matching
            full_tensor_name = f"blk.{layer_idx}.{tensor_type}"
            
            # Parse the tensor override pattern
            # Example patterns:
            # blk\.(?:[0-9][02468]|[0-9][159])\.ffn.*_exps\.=CPU
            # blk\.[0-9]+\.attn_.*=CPU
            # blk\.(?:1[0-9]|2[0-9])\..*=CPU
            
            # Extract the regex pattern part (before =)
            if '=' in self.tensor_pattern:
                pattern_part = self.tensor_pattern.split('=')[0]
            else:
                pattern_part = self.tensor_pattern
                
            # Try to match the pattern
            try:
                if re.match(pattern_part, full_tensor_name):
                    return True
            except re.error:
                # If regex fails, try simpler pattern matching
                pass
                
            # Fallback to simpler pattern matching
            if "blk\\." in pattern_part:
                # Extract layer number patterns
                if "[02468]" in pattern_part and layer_idx % 2 == 0:
                    # Even layers
                    if "ffn" in pattern_part and "ffn" in tensor_type:
                        return True
                    elif "attn" in pattern_part and "attn" in tensor_type:
                        return True
                elif "[159]" in pattern_part and layer_idx % 2 == 1:
                    # Odd layers  
                    if "ffn" in pattern_part and "ffn" in tensor_type:
                        return True
                    elif "attn" in pattern_part and "attn" in tensor_type:
                        return True
                        
                # Range patterns like [1-9] or [10-20]
                range_match = re.search(r'\[(\d+)-(\d+)\]', pattern_part)
                if range_match:
                    start_layer = int(range_match.group(1))
                    end_layer = int(range_match.group(2))
                    if start_layer <= layer_idx <= end_layer:
                        if "ffn" in pattern_part and "ffn" in tensor_type:
                            return True
                        elif "attn" in pattern_part and "attn" in tensor_type:
                            return True
                            
            return False
            
        except Exception:
            return False
            
    def _draw_legend(self, painter, rect):
        """Draw legend with pattern information"""
        # Draw legend background to ensure text is readable
        legend_start_y = rect.height() - 45
        legend_rect = (0, legend_start_y - 5, rect.width(), 45)
        painter.fillRect(*legend_rect, QColor(40, 40, 40, 200))  # Semi-transparent background
        
        painter.setPen(QColor(255, 255, 255))
        
        # Basic stats (top line)
        current_y = legend_start_y + 10
        pattern_affected = sum(1 for i in range(self.total_layers) 
                             for tensor_type in self.layer_tensor_map.get(i, [])
                             if self._matches_tensor_pattern(i, tensor_type))
        
        stats_text = f"Layers: {self.total_layers} | GPU: {self.gpu_layers} | CPU: {self.total_layers - self.gpu_layers}"
        if pattern_affected > 0:
            stats_text += f" | Overridden: {pattern_affected} tensors"
            
        painter.drawText(10, current_y, stats_text)
        
        # Color legend (middle line)
        current_y += 15
        legend_items = [
            (QColor(76, 175, 80), "GPU"),
            (QColor(244, 67, 54), "CPU")
        ]
        
        x_offset = 10
        for color, label in legend_items:
            painter.fillRect(x_offset, current_y - 8, 12, 12, color)
            painter.drawText(x_offset + 16, current_y + 2, label)
            x_offset += 80
        
        # Pattern info (bottom line)
        current_y += 15
        if self.tensor_pattern:
            pattern_text = f"Pattern: {self.tensor_pattern[:50]}{'...' if len(self.tensor_pattern) > 50 else ''}"
            painter.drawText(10, current_y, pattern_text)
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement for tooltips"""
        pos = event.position().toPoint()
        
        # Check if mouse is over a layer block
        for layer_idx, (x, y, width, height) in self.layer_blocks.items():
            if x <= pos.x() <= x + width and y <= pos.y() <= y + height:
                self._show_layer_tooltip(layer_idx, pos)
                return
                
        # Clear tooltip if not over any block
        self.setToolTip("")
        
    def _show_layer_tooltip(self, layer_idx, pos):
        """Show detailed tooltip for a layer"""
        tensor_types = self.layer_tensor_map.get(layer_idx, self.common_tensor_types)
        
        # Build tooltip text
        tooltip_lines = [f"Layer {layer_idx}"]
        
        if layer_idx < self.gpu_layers:
            tooltip_lines.append("Base allocation: GPU")
        else:
            tooltip_lines.append("Base allocation: CPU")
            
        tooltip_lines.append(f"Tensors ({len(tensor_types)}):")
        
        # Show tensor allocation details
        for tensor_type in tensor_types[:8]:  # Limit for readability
            color_info = self._get_tensor_color(layer_idx, tensor_type)
            is_overridden = self.tensor_pattern and self._matches_tensor_pattern(layer_idx, tensor_type)
            
            if color_info == QColor(76, 175, 80):  # GPU color
                allocation = "GPU (Override)" if is_overridden else "GPU"
            else:  # CPU color
                allocation = "CPU (Override)" if is_overridden else "CPU"
                
            tooltip_lines.append(f"  • {tensor_type}: {allocation}")
            
        if len(tensor_types) > 8:
            tooltip_lines.append(f"  ... and {len(tensor_types) - 8} more")
            
        # Add pattern match info
        if self.tensor_pattern:
            affected_tensors = sum(1 for t in tensor_types 
                                 if self._matches_tensor_pattern(layer_idx, t))
            if affected_tensors > 0:
                tooltip_lines.append(f"Pattern affects: {affected_tensors} tensors")
                
        self.setToolTip("\n".join(tooltip_lines))

class GGUFReader:
    """Optimized GGUF file reader - only reads essential tensor information"""
    
    GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def read_string(self, f) -> str:
        """Read a string from GGUF file"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
        
    def skip_value(self, f, value_type: int):
        """Skip a value based on its type without reading it into memory"""
        if value_type == 0 or value_type == 1:  # uint8, int8
            f.read(1)
        elif value_type == 2 or value_type == 3:  # uint16, int16
            f.read(2)
        elif value_type == 4 or value_type == 5:  # uint32, int32
            f.read(4)
        elif value_type == 6:  # float32
            f.read(4)
        elif value_type == 7:  # bool
            f.read(1)
        elif value_type == 8:  # string
            length = struct.unpack('<Q', f.read(8))[0]
            f.read(length)  # Skip string content
        elif value_type == 10 or value_type == 11:  # uint64, int64
            f.read(8)
        elif value_type == 12:  # float64
            f.read(8)
        elif value_type == 9:  # array - skip for now
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            # Skip array elements based on type
            for _ in range(array_len):
                self.skip_value(f, array_type)
        
    def read_gguf_info(self) -> Dict:
        """Read GGUF file and extract only essential tensor information"""
        try:
            with open(self.file_path, 'rb') as f:
                # Read header
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != self.GGUF_MAGIC:
                    return {'error': 'Not a valid GGUF file'}
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
                
                # Skip metadata efficiently - only look for model name
                model_name = "Unknown"
                
                for _ in range(metadata_kv_count):
                    key = self.read_string(f)
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    if key == "general.name" and value_type == 8:  # string
                        model_name = self.read_string(f)
                    else:
                        # Skip all other metadata values
                        self.skip_value(f, value_type)
                
                # Read tensor names efficiently - collect representative samples
                block_numbers = set()
                sample_tensor_names = []
                layer_tensor_samples = {}  # Store samples per layer for better visualization
                
                for i in range(tensor_count):
                    tensor_name = self.read_string(f)
                    
                    # Keep diverse tensor names for display and analysis
                    if len(sample_tensor_names) < 20:
                        sample_tensor_names.append(tensor_name)
                    
                    # Extract block/layer number from tensor name
                    match = re.search(r'blk\.(\d+)', tensor_name)
                    if match:
                        layer_idx = int(match.group(1))
                        block_numbers.add(layer_idx)
                        
                        # Collect diverse tensor types per layer (for first few layers)
                        if layer_idx < 3 and layer_idx not in layer_tensor_samples:
                            layer_tensor_samples[layer_idx] = []
                        if layer_idx < 3 and len(layer_tensor_samples[layer_idx]) < 10:
                            layer_tensor_samples[layer_idx].append(tensor_name)
                    
                    # Skip tensor dimensions and type info without reading
                    n_dims = struct.unpack('<I', f.read(4))[0]
                    f.seek(f.tell() + n_dims * 8)  # Skip dimensions
                    f.seek(f.tell() + 4)  # Skip type
                    f.seek(f.tell() + 8)  # Skip offset
                
                # Enhance sample tensor names with layer samples
                all_tensor_samples = sample_tensor_names.copy()
                for layer_samples in layer_tensor_samples.values():
                    all_tensor_samples.extend(layer_samples)
                
                # Calculate actual layers
                if block_numbers:
                    actual_layers = max(block_numbers) + 1  # 0-indexed
                else:
                    # Fallback: estimate based on tensor count
                    actual_layers = max(1, tensor_count // 10)
                
                return {
                    'version': version,
                    'tensor_count': tensor_count,
                    'metadata_count': metadata_kv_count,
                    'estimated_layers': actual_layers,
                    'model_name': model_name,
                    'tensor_names': all_tensor_samples,
                    'block_count': len(block_numbers) if block_numbers else 0,
                    'layer_tensor_samples': layer_tensor_samples
                }
                
        except Exception as e:
            # Fallback to simple estimation if detailed parsing fails
            try:
                stat = os.stat(self.file_path)
                size_mb = stat.st_size / (1024 * 1024)
                estimated_layers = max(1, int(size_mb / 100))
                
                return {
                    'version': 'Unknown',
                    'tensor_count': 0,
                    'metadata_count': 0,
                    'estimated_layers': estimated_layers,
                    'model_name': 'Unknown',
                    'error': f'Partial read error: {str(e)}'
                }
            except:
                return {'error': f'Failed to read GGUF: {str(e)}'}

class ModelScanThread(QThread):
    """Background thread for scanning GGUF models"""
    
    models_found = pyqtSignal(list)
    progress = pyqtSignal(int, str)
    
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        
    def run(self):
        """Scan directory for GGUF files - only basic file info, no GGUF parsing"""
        try:
            models = []
            if not os.path.exists(self.directory):
                self.models_found.emit([])
                return
                
            # Find all GGUF files recursively
            pattern = os.path.join(self.directory, "**", "*.gguf")
            files = glob.glob(pattern, recursive=True)
            
            total_files = len(files)
            for i, file_path in enumerate(files):
                self.progress.emit(int((i / total_files) * 100), f"Scanning: {os.path.basename(file_path)}")
                
                # Get basic file info only
                stat = os.stat(file_path)
                size_mb = stat.st_size / (1024 * 1024)
                
                models.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size_mb': size_mb,
                    'directory': os.path.dirname(file_path),
                    'gguf_info': None  # Will be loaded on demand when model is selected
                })
                
            self.models_found.emit(models)
            
        except Exception as e:
            print(f"Error scanning models: {e}")
            self.models_found.emit([])

class GGUFAnalysisThread(QThread):
    """Background thread for analyzing a single GGUF file when selected"""
    
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        """Analyze the selected GGUF file"""
        try:
            reader = GGUFReader(self.file_path)
            gguf_info = reader.read_gguf_info()
            self.analysis_complete.emit(gguf_info)
        except Exception as e:
            self.analysis_complete.emit({'error': f'Analysis failed: {str(e)}'})

class ParameterWidget(QWidget):
    """Widget for editing llama.cpp parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.launcher_ref = None  # Reference to main launcher for updates
        self.setup_ui()
        
    def setup_ui(self):
        """Setup parameter editing interface"""
        layout = QVBoxLayout(self)
        
        # Create tabs for different parameter categories
        tabs = QTabWidget()
        
        # Model & Context tab
        model_tab = self.create_model_tab()
        tabs.addTab(model_tab, "Model & Context")
        
        # Performance tab
        perf_tab = self.create_performance_tab()
        tabs.addTab(perf_tab, "Performance")
        
        # Sampling tab
        sampling_tab = self.create_sampling_tab()
        tabs.addTab(sampling_tab, "Sampling")
        
        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tabs)
        
    def create_model_tab(self):
        """Create model and context parameters tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Context size
        layout.addWidget(QLabel("Context Size:"), 0, 0)
        self.ctx_size = QSpinBox()
        self.ctx_size.setRange(512, 131072)
        self.ctx_size.setValue(4096)
        self.ctx_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.ctx_size, 0, 1)
        
        # Batch size
        layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 8192)
        self.batch_size.setValue(2048)
        self.batch_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.batch_size, 1, 1)
        
        # Micro batch size
        layout.addWidget(QLabel("Micro Batch Size:"), 2, 0)
        self.ubatch_size = QSpinBox()
        self.ubatch_size.setRange(1, 2048)
        self.ubatch_size.setValue(512)
        self.ubatch_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.ubatch_size, 2, 1)
        
        # System prompt
        layout.addWidget(QLabel("System Prompt:"), 3, 0)
        self.system_prompt = QLineEdit()
        self.system_prompt.setPlaceholderText("You are a helpful assistant")
        layout.addWidget(self.system_prompt, 3, 1)
        
        # Chat template
        layout.addWidget(QLabel("Chat Template:"), 4, 0)
        self.chat_template = QComboBox()
        self.chat_template.addItems([
            "auto", "llama3", "chatml", "mistral-v1", "mistral-v3", 
            "gemma", "phi3", "deepseek", "command-r"
        ])
        layout.addWidget(self.chat_template, 4, 1)
        
        return widget
        
    def create_performance_tab(self):
        """Create performance parameters tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # GPU layers
        layout.addWidget(QLabel("GPU Layers:"), 0, 0)
        self.n_gpu_layers = QSpinBox()
        self.n_gpu_layers.setRange(0, 999)
        self.n_gpu_layers.setValue(0)
        self.n_gpu_layers.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.n_gpu_layers, 0, 1)
        
        # Threads
        layout.addWidget(QLabel("Threads:"), 1, 0)
        self.threads = QSpinBox()
        self.threads.setRange(-1, 64)
        self.threads.setValue(-1)
        self.threads.setSpecialValueText("Auto")
        self.threads.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.threads, 1, 1)
        
        # Flash attention
        self.flash_attn = QCheckBox("Enable Flash Attention")
        layout.addWidget(self.flash_attn, 2, 0, 1, 2)
        
        # Tensor override pattern
        layout.addWidget(QLabel("Tensor Override:"), 3, 0)
        self.tensor_override = QLineEdit()
        self.tensor_override.setPlaceholderText('blk\\.(?:[0-9][02468]|[0-9][159])\\.ffn.*_exps\\.=CPU')
        self.tensor_override.textChanged.connect(self.on_tensor_override_changed)
        layout.addWidget(self.tensor_override, 3, 1)
        
        # Add pattern examples button
        pattern_examples_btn = QPushButton("Examples")
        pattern_examples_btn.clicked.connect(self.show_pattern_examples)
        layout.addWidget(pattern_examples_btn, 3, 2)
        
        return widget
        
    def create_sampling_tab(self):
        """Create sampling parameters tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Temperature
        layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.8)
        self.temperature.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.temperature, 0, 1)
        
        # Top-K
        layout.addWidget(QLabel("Top-K:"), 1, 0)
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 200)
        self.top_k.setValue(40)
        self.top_k.setSpecialValueText("Disabled")
        self.top_k.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.top_k, 1, 1)
        
        # Top-P
        layout.addWidget(QLabel("Top-P:"), 2, 0)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.9)
        self.top_p.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.top_p, 2, 1)
        
        # Repeat penalty
        layout.addWidget(QLabel("Repeat Penalty:"), 3, 0)
        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(0.0, 2.0)
        self.repeat_penalty.setSingleStep(0.05)
        self.repeat_penalty.setValue(1.0)
        self.repeat_penalty.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        layout.addWidget(self.repeat_penalty, 3, 1)
        
        return widget
        
    def create_advanced_tab(self):
        """Create advanced parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Server options group
        server_group = QGroupBox("Server Options")
        server_layout = QGridLayout(server_group)
        
        # Host
        server_layout.addWidget(QLabel("Host:"), 0, 0)
        self.server_host = QLineEdit()
        self.server_host.setText("127.0.0.1")
        self.server_host.setPlaceholderText("127.0.0.1")
        server_layout.addWidget(self.server_host, 0, 1)
        
        # Port
        server_layout.addWidget(QLabel("Port:"), 1, 0)
        self.server_port = QSpinBox()
        self.server_port.setRange(1, 65535)
        self.server_port.setValue(8080)
        self.server_port.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        server_layout.addWidget(self.server_port, 1, 1)
        
        layout.addWidget(server_group)
        
        # Additional command line arguments
        layout.addWidget(QLabel("Additional Arguments:"))
        self.additional_args = QTextEdit()
        self.additional_args.setMaximumHeight(100)
        self.additional_args.setPlaceholderText("--mlock --no-mmap")
        layout.addWidget(self.additional_args)
        
        return widget
        
    def get_config(self) -> ModelConfig:
        """Get current configuration"""
        return ModelConfig(
            ctx_size=self.ctx_size.value(),
            n_gpu_layers=self.n_gpu_layers.value(),
            batch_size=self.batch_size.value(),
            ubatch_size=self.ubatch_size.value(),
            threads=self.threads.value(),
            flash_attn=self.flash_attn.isChecked(),
            tensor_override=self.tensor_override.text(),
            temperature=self.temperature.value(),
            top_k=self.top_k.value(),
            top_p=self.top_p.value(),
            repeat_penalty=self.repeat_penalty.value(),
            system_prompt=self.system_prompt.text(),
            chat_template=self.chat_template.currentText(),
            server_host=self.server_host.text(),
            server_port=self.server_port.value()
        )
        
    def set_config(self, config: ModelConfig):
        """Set configuration values"""
        self.ctx_size.setValue(config.ctx_size)
        self.n_gpu_layers.setValue(config.n_gpu_layers)
        self.batch_size.setValue(config.batch_size)
        self.ubatch_size.setValue(config.ubatch_size)
        self.threads.setValue(config.threads)
        self.flash_attn.setChecked(config.flash_attn)
        self.tensor_override.setText(config.tensor_override)
        self.temperature.setValue(config.temperature)
        self.top_k.setValue(config.top_k)
        self.top_p.setValue(config.top_p)
        self.repeat_penalty.setValue(config.repeat_penalty)
        self.system_prompt.setText(config.system_prompt)
        self.server_host.setText(config.server_host)
        self.server_port.setValue(config.server_port)
        
        # Set chat template
        index = self.chat_template.findText(config.chat_template)
        if index >= 0:
            self.chat_template.setCurrentIndex(index)
            
    def on_tensor_override_changed(self):
        """Handle tensor override pattern change"""
        # Use direct reference if available
        if self.launcher_ref and hasattr(self.launcher_ref, 'update_tensor_visualization'):
            self.launcher_ref.update_tensor_visualization()
            return
            
        # Fallback: search through parent hierarchy
        parent = self.parent()
        while parent and not hasattr(parent, 'update_tensor_visualization'):
            parent = parent.parent()
        if parent and hasattr(parent, 'update_tensor_visualization'):
            parent.update_tensor_visualization()
            
    def show_pattern_examples(self):
        """Show dialog with tensor override pattern examples"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Tensor Override Pattern Examples")
        dialog.setModal(True)
        dialog.resize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Create text area with examples
        examples_text = QTextEdit()
        examples_text.setReadOnly(True)
        examples_text.setPlainText("""
Common Tensor Override Patterns:

1. Offload FFN layers of even-numbered blocks to CPU:
   blk\\.(?:[0-9][02468]|[0-9][159])\\.ffn.*=CPU

2. Offload attention layers of specific range to CPU:
   blk\\.[1-9]\\.attn_.*=CPU

3. Offload all tensors in layers 10-20 to CPU:
   blk\\.(?:1[0-9]|20)\\..*=CPU

4. Offload only FFN gate and up projections:
   blk\\.[0-9]+\\.ffn_(?:gate|up).*=CPU

5. Offload attention output projections:
   blk\\.[0-9]+\\.attn_output.*=CPU

6. Complex pattern - FFN experts in even layers:
   blk\\.(?:[0-9][02468])\\.ffn.*_exps\\..*=CPU

7. Offload normalization layers:
   blk\\.[0-9]+\\.(?:attn_norm|ffn_norm).*=CPU

8. Offload specific layer range (layers 5-15):
   blk\\.(?:[5-9]|1[0-5])\\..*=CPU

Pattern Syntax:
- blk\\. matches "blk." literally
- [0-9] matches any digit
- [02468] matches even digits
- [159] matches odd digits (1,5,9)
- .* matches any characters
- (?:...) is a non-capturing group
- | means OR
- =CPU specifies CPU allocation
- =GPU specifies GPU allocation (if supported)

Tips:
- Test patterns carefully as they affect performance
- Start with simple patterns and gradually add complexity
- Use the visualization to verify pattern effects
- Patterns are case-sensitive
        """)
        layout.addWidget(examples_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Quick insert buttons for common patterns
        patterns = [
            ("Even FFN → CPU", "blk\\.(?:[0-9][02468])\\.ffn.*=CPU"),
            ("Odd FFN → CPU", "blk\\.(?:[0-9][159])\\.ffn.*=CPU"),
            ("Attn layers 1-9 → CPU", "blk\\.[1-9]\\.attn_.*=CPU"),
            ("All layers 10+ → CPU", "blk\\.(?:[1-9][0-9]|[1-9][0-9][0-9])\\..*=CPU")
        ]
        
        for label, pattern in patterns:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, p=pattern: self._insert_pattern(p))
            button_layout.addWidget(btn)
            
        layout.addLayout(button_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
        
    def _insert_pattern(self, pattern):
        """Insert a pattern into the tensor override field"""
        self.tensor_override.setText(pattern)
        self.tensor_override.parent().close()  # Close the dialog

class LlamaLauncher(QMainWindow):
    """Main launcher window"""
    
    def __init__(self):
        super().__init__()
        self.models = []
        self.current_model = None
        self.global_config = GlobalConfig()
        self.model_configs = {}
        
        self.setup_ui()
        self.apply_dark_theme()
        self.load_settings()
        
        # Auto-scan models directory if it exists
        if self.global_config.models_directory and os.path.exists(self.global_config.models_directory):
            QTimer.singleShot(500, self.scan_models)  # Delay to let UI finish loading
        
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Llama.cpp Enhanced Launcher")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Model selection
        left_panel = self.create_model_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Parameters and visualization
        right_panel = self.create_parameter_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_model_panel(self):
        """Create the model selection panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Directory selection
        dir_group = QGroupBox("Models Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self.models_dir_edit = QLineEdit()
        self.models_dir_edit.setPlaceholderText("Select models directory...")
        dir_layout.addWidget(self.models_dir_edit)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_models_directory)
        dir_layout.addWidget(browse_btn)
        
        scan_btn = QPushButton("Scan")
        scan_btn.clicked.connect(self.scan_models)
        dir_layout.addWidget(scan_btn)
        
        layout.addWidget(dir_group)
        
        # Model list
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout(models_group)
        
        # Search/filter
        self.model_filter = QLineEdit()
        self.model_filter.setPlaceholderText("Filter models...")
        self.model_filter.textChanged.connect(self.filter_models)
        models_layout.addWidget(self.model_filter)
        
        # Model list widget
        self.model_list = QListWidget()
        self.model_list.itemSelectionChanged.connect(self.on_model_selected)
        models_layout.addWidget(self.model_list)
        
        # Progress bar for scanning
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        models_layout.addWidget(self.scan_progress)
        
        layout.addWidget(models_group)
        
        # Executable selection
        exe_group = QGroupBox("Executable")
        exe_layout = QVBoxLayout(exe_group)
        
        exe_select_layout = QHBoxLayout()
        self.executable_edit = QLineEdit()
        self.executable_edit.setPlaceholderText("Path to llama folder")
        exe_select_layout.addWidget(self.executable_edit)
        
        exe_browse_btn = QPushButton("Browse")
        exe_browse_btn.clicked.connect(self.browse_executable)
        exe_select_layout.addWidget(exe_browse_btn)
        
        exe_layout.addLayout(exe_select_layout)
        
        # Mode selection - dropdown
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["llama-server.exe", "llama-cli.exe"])
        self.mode_combo.setCurrentText("llama-server.exe")
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()  # Push combo to the left
        
        exe_layout.addLayout(mode_layout)
        
        layout.addWidget(exe_group)
        
        return panel
        
    def create_parameter_panel(self):
        """Create the parameter configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setWordWrap(True)
        info_layout.addWidget(self.model_info_label)
        
        layout.addWidget(info_group)
        
        # Tensor visualization
        viz_group = QGroupBox("Tensor Allocation Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.tensor_viz = TensorVisualizationWidget()
        viz_layout.addWidget(self.tensor_viz)
        
        # GPU layers slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("GPU Layers:"))
        
        self.gpu_layers_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_layers_slider.setRange(0, 100)
        self.gpu_layers_slider.valueChanged.connect(self.on_gpu_layers_changed)
        slider_layout.addWidget(self.gpu_layers_slider)
        
        self.gpu_layers_label = QLabel("0")
        slider_layout.addWidget(self.gpu_layers_label)
        
        viz_layout.addLayout(slider_layout)
        layout.addWidget(viz_group)
        
        # Parameters
        self.parameter_widget = ParameterWidget()
        # Set reference to main launcher for direct updates
        self.parameter_widget.launcher_ref = self
        # Connect parameter changes to visualization updates
        self.parameter_widget.n_gpu_layers.valueChanged.connect(self.on_parameter_changed)
        # Connect tensor override changes with a small delay to avoid excessive updates
        self.tensor_update_timer = QTimer()
        self.tensor_update_timer.setSingleShot(True)
        self.tensor_update_timer.timeout.connect(self.update_tensor_visualization)
        self.parameter_widget.tensor_override.textChanged.connect(self.on_tensor_override_text_changed)
        layout.addWidget(self.parameter_widget)
        
        # Launch buttons
        button_layout = QHBoxLayout()
        
        self.launch_btn = QPushButton("Launch")
        self.launch_btn.clicked.connect(self.launch_model)
        self.launch_btn.setEnabled(False)
        button_layout.addWidget(self.launch_btn)
        
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_model_config)
        button_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_model_config)
        button_layout.addWidget(self.load_config_btn)
        
        layout.addLayout(button_layout)
        
        return panel     
   
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #0078d4;
            }
            
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 3px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            
            QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                alternate-background-color: #404040;
            }
            
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #555555;
            }
            
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            
            QListWidget::item:hover {
                background-color: #404040;
            }
            
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
            
            QTabBar::tab {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 8px 16px;
                margin-right: 2px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            
            QTabBar::tab:hover {
                background-color: #404040;
            }
            
            QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
            }
            
            QCheckBox {
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                background-color: #3c3c3c;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                border-radius: 3px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3c3c3c;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
            
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
            }
            
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            
            QStatusBar {
                background-color: #3c3c3c;
                border-top: 1px solid #555555;
            }
            
            QSplitter::handle {
                background-color: #555555;
            }
            
            QSplitter::handle:horizontal {
                width: 3px;
            }
            
            QSplitter::handle:vertical {
                height: 3px;
            }
        """)
        
    def browse_models_directory(self):
        """Browse for models directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", self.models_dir_edit.text()
        )
        if directory:
            self.models_dir_edit.setText(directory)
            self.global_config.models_directory = directory
            
    def browse_executable(self):
        """Browse for llama folder containing executables"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Llama Folder", 
            self.executable_edit.text() or ""
        )
        if folder_path:
            self.executable_edit.setText(folder_path)
            self.global_config.executable_path = folder_path
            
    def scan_models(self):
        """Start scanning for models in the selected directory"""
        directory = self.models_dir_edit.text()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "Warning", "Please select a valid models directory first.")
            return
            
        self.scan_progress.setVisible(True)
        self.scan_progress.setValue(0)
        self.statusBar().showMessage("Scanning for models...")
        
        # Start background scanning thread
        self.scan_thread = ModelScanThread(directory)
        self.scan_thread.models_found.connect(self.on_models_found)
        self.scan_thread.progress.connect(self.on_scan_progress)
        self.scan_thread.start()
        
    def on_scan_progress(self, value: int, message: str):
        """Update scan progress"""
        self.scan_progress.setValue(value)
        self.statusBar().showMessage(message)
        
    def on_models_found(self, models: List[Dict]):
        """Handle found models"""
        self.models = models
        self.update_model_list()
        self.scan_progress.setVisible(False)
        self.statusBar().showMessage(f"Found {len(models)} models")
        
        # Auto-select the last selected model if it exists
        self.auto_select_last_model()
        
    def update_model_list(self):
        """Update the model list widget"""
        self.model_list.clear()
        filter_text = self.model_filter.text().lower()
        
        for model in self.models:
            if not filter_text or filter_text in model['name'].lower():
                item = QListWidgetItem()
                item.setText(f"{model['name']}\n{model['size_mb']:.1f} MB")
                item.setData(Qt.ItemDataRole.UserRole, model)
                self.model_list.addItem(item)
                
    def filter_models(self):
        """Filter models based on search text"""
        self.update_model_list()
        
    def auto_select_last_model(self):
        """Auto-select the last selected model if it exists in the current model list"""
        if not self.global_config.last_model:
            return
            
        # Find the model in the current list
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            model = item.data(Qt.ItemDataRole.UserRole)
            if model['path'] == self.global_config.last_model:
                # Select the item and trigger the selection event
                self.model_list.setCurrentItem(item)
                self.statusBar().showMessage(f"Auto-selected last model: {model['name']}")
                break
        
    def on_model_selected(self):
        """Handle model selection"""
        current_item = self.model_list.currentItem()
        if not current_item:
            return
            
        model = current_item.data(Qt.ItemDataRole.UserRole)
        self.current_model = model
        
        # Save the last selected model path
        self.global_config.last_model = model['path']
        # Save settings immediately to persist the last model selection
        self.save_settings()
        
        # Show basic info immediately
        info_text = f"""
        <b>Name:</b> {model['name']}<br>
        <b>Size:</b> {model['size_mb']:.1f} MB<br>
        <b>Path:</b> {model['path']}<br>
        <b>Directory:</b> {model['directory']}<br>
        <b>Status:</b> Analyzing GGUF file...
        """
        self.model_info_label.setText(info_text)
        
        # Use fallback estimation initially
        estimated_layers = max(1, int(model['size_mb'] / 100))
        self.gpu_layers_slider.setMaximum(estimated_layers)
        
        # Update tensor visualization with estimation
        gpu_layers = self.parameter_widget.n_gpu_layers.value()
        pattern = self.parameter_widget.tensor_override.text()
        self.tensor_viz.set_layers(estimated_layers, gpu_layers, pattern, [])
        
        # Load model-specific config if exists
        self.load_model_config_if_exists(model['path'])
        
        # Enable launch button
        self.launch_btn.setEnabled(True)
        
        # Start GGUF analysis in background if not already done
        if model.get('gguf_info') is None:
            self.analyze_gguf_file(model['path'])
        else:
            # Use cached GGUF info
            self.update_model_info_with_gguf(model['gguf_info'])
            
    def analyze_gguf_file(self, file_path: str):
        """Start background GGUF analysis"""
        self.statusBar().showMessage("Analyzing GGUF file...")
        
        # Start analysis thread
        self.gguf_thread = GGUFAnalysisThread(file_path)
        self.gguf_thread.analysis_complete.connect(self.on_gguf_analysis_complete)
        self.gguf_thread.start()
        
    def on_gguf_analysis_complete(self, gguf_info: Dict):
        """Handle completed GGUF analysis"""
        if not self.current_model:
            return
            
        # Cache the GGUF info in the model
        self.current_model['gguf_info'] = gguf_info
        
        # Update the model info display
        self.update_model_info_with_gguf(gguf_info)
        
        # Update tensor visualization with actual layer count
        if 'error' not in gguf_info and 'estimated_layers' in gguf_info:
            total_layers = gguf_info['estimated_layers']
            self.gpu_layers_slider.setMaximum(total_layers)
            
            gpu_layers = self.parameter_widget.n_gpu_layers.value()
            pattern = self.parameter_widget.tensor_override.text()
            tensor_names = gguf_info.get('tensor_names', [])
            
            self.tensor_viz.set_layers(total_layers, gpu_layers, pattern, tensor_names)
        
        self.statusBar().showMessage("GGUF analysis complete")
        
    def update_model_info_with_gguf(self, gguf_info: Dict):
        """Update model info display with GGUF data"""
        if not self.current_model:
            return
            
        model = self.current_model
        
        # Update model info - simplified without path/directory details
        info_text = f"""
        <b>Name:</b> {model['name']}<br>
        <b>Size:</b> {model['size_mb']:.1f} MB<br>
        """
        
        if 'error' not in gguf_info:
            info_text += f"""
            <b>Model Name:</b> {gguf_info.get('model_name', 'Unknown')}<br>
            <b>GGUF Version:</b> {gguf_info.get('version', 'Unknown')}<br>
            <b>Tensor Count:</b> {gguf_info.get('tensor_count', 0)}<br>
            <b>Layers:</b> {gguf_info.get('estimated_layers', 0)}<br>
            <b>Block Count:</b> {gguf_info.get('block_count', 0)}
            """
                    
        else:
            info_text += f"<b>GGUF Error:</b> {gguf_info['error']}"
            
        self.model_info_label.setText(info_text)
        
    def update_tensor_visualization(self):
        """Update tensor visualization when pattern changes"""
        if self.current_model:
            gguf_info = self.current_model.get('gguf_info')
            if gguf_info and 'error' not in gguf_info and 'estimated_layers' in gguf_info:
                total_layers = gguf_info['estimated_layers']
                tensor_names = gguf_info.get('tensor_names', [])
            else:
                total_layers = max(1, int(self.current_model['size_mb'] / 100))
                tensor_names = []
            
            gpu_layers = self.parameter_widget.n_gpu_layers.value()
            pattern = self.parameter_widget.tensor_override.text()
            
            self.tensor_viz.set_layers(total_layers, gpu_layers, pattern, tensor_names)
        
    def on_gpu_layers_changed(self, value: int):
        """Handle GPU layers slider change"""
        self.gpu_layers_label.setText(str(value))
        self.parameter_widget.n_gpu_layers.setValue(value)
        self.update_tensor_visualization()
        
    def on_parameter_changed(self):
        """Handle any parameter change that affects visualization"""
        # Sync slider with spinbox
        gpu_layers = self.parameter_widget.n_gpu_layers.value()
        if self.gpu_layers_slider.value() != gpu_layers:
            self.gpu_layers_slider.setValue(gpu_layers)
            self.gpu_layers_label.setText(str(gpu_layers))
        
        # Update visualization
        self.update_tensor_visualization()
        
    def on_tensor_override_text_changed(self):
        """Handle tensor override text changes with debouncing"""
        # Restart the timer - this debounces rapid typing
        self.tensor_update_timer.start(300)  # 300ms delay
        

    def build_command_line(self) -> List[str]:
        """Build command line arguments"""
        if not self.current_model or not self.executable_edit.text():
            return []
            
        config = self.parameter_widget.get_config()
        
        # Use the selected mode from combo box
        base_path = self.executable_edit.text()
        selected_mode = self.mode_combo.currentText()
        
        # Construct the full executable path from the folder
        executable_path = os.path.join(base_path, selected_mode)
            
        cmd = [executable_path]
        
        # Model path - always quote the path
        cmd.extend(["-m", f'"{self.current_model["path"]}"'])
        
        # Context size
        if config.ctx_size != 4096:
            cmd.extend(["-c", str(config.ctx_size)])
            
        # GPU layers
        if config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(config.n_gpu_layers)])
            
        # Batch sizes
        if config.batch_size != 2048:
            cmd.extend(["-b", str(config.batch_size)])
        if config.ubatch_size != 512:
            cmd.extend(["-ub", str(config.ubatch_size)])
            
        # Threads
        if config.threads != -1:
            cmd.extend(["-t", str(config.threads)])
            
        # Flash attention
        if config.flash_attn:
            cmd.append("-fa")
            
        # Tensor override - quote if contains spaces or special characters
        if config.tensor_override.strip():
            override = config.tensor_override.strip()
            if " " in override or any(c in override for c in ['(', ')', '[', ']', '|', '&']):
                cmd.extend(["-ot", f'"{override}"'])
            else:
                cmd.extend(["-ot", override])
            
        # Sampling parameters
        if config.temperature != 0.8:
            cmd.extend(["--temp", str(config.temperature)])
        if config.top_k != 40:
            cmd.extend(["--top-k", str(config.top_k)])
        if config.top_p != 0.9:
            cmd.extend(["--top-p", str(config.top_p)])
        if config.repeat_penalty != 1.0:
            cmd.extend(["--repeat-penalty", str(config.repeat_penalty)])
            
        # System prompt - only for CLI mode, not server mode
        selected_mode = self.mode_combo.currentText()
        if selected_mode == "llama-cli.exe" and config.system_prompt.strip():
            prompt = config.system_prompt.strip()
            if " " in prompt:
                cmd.extend(["-sys", f'"{prompt}"'])
            else:
                cmd.extend(["-sys", prompt])
        
        # Server-specific parameters
        if selected_mode == "llama-server.exe":
            cmd.extend(["--host", config.server_host])
            cmd.extend(["--port", str(config.server_port)])
            
        # Chat template
        if config.chat_template and config.chat_template != "auto":
            cmd.extend(["--chat-template", config.chat_template])
            
        # Additional arguments
        additional = getattr(self.parameter_widget, 'additional_args', None)
        if additional and hasattr(additional, 'toPlainText'):
            extra_args = additional.toPlainText().strip()
            if extra_args:
                cmd.extend(extra_args.split())
                
        return cmd
        
    def launch_model(self):
        """Launch the model with current configuration"""
        cmd = self.build_command_line()
        if not cmd:
            QMessageBox.warning(self, "Warning", "Cannot build command line. Check model and executable selection.")
            return
            
        # Show command preview dialog
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Launch Command Preview")
        preview_dialog.setModal(True)
        preview_dialog.resize(800, 400)
        
        layout = QVBoxLayout(preview_dialog)
        
        layout.addWidget(QLabel("Command to execute in new terminal:"))
        
        cmd_text = QTextEdit()
        cmd_text.setPlainText(" ".join(cmd))
        cmd_text.setReadOnly(True)
        layout.addWidget(cmd_text)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(preview_dialog.accept)
        buttons.rejected.connect(preview_dialog.reject)
        layout.addWidget(buttons)
        
        if preview_dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                # Launch in new terminal window
                self.launch_in_terminal(cmd)
                self.statusBar().showMessage("Model launched in new terminal!")
                
                # Save current configuration
                self.save_model_config()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to launch model:\n{str(e)}")
                
    def launch_in_terminal(self, cmd: List[str]):
        """Launch command in a new terminal window"""
        exe_dir = os.path.dirname(self.executable_edit.text())
        
        # Build command string for terminal - handle already quoted arguments
        def quote_arg(arg):
            # If argument is already quoted, don't double-quote
            if arg.startswith('"') and arg.endswith('"'):
                return arg
            # Quote if contains spaces or special characters
            elif " " in arg or any(c in arg for c in ['(', ')', '[', ']', '|', '&', '<', '>']):
                return f'"{arg}"'
            else:
                return arg
        
        cmd_str = " ".join(quote_arg(arg) for arg in cmd)
        
        # Windows-specific terminal launch
        if sys.platform == "win32":
            try:
                # Create a simple batch command
                batch_content = f'@echo off\ncd /d "{exe_dir}"\necho Starting llama.cpp...\necho.\n{cmd_str}\necho.\necho Process finished.\npause'
                
                # Write to temp file and execute
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
                    f.write(batch_content)
                    temp_bat = f.name
                
                # Launch the batch file
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/c', temp_bat])
                
                # Clean up temp file after a delay
                import threading
                def cleanup():
                    import time
                    time.sleep(5)  # Wait 5 seconds
                    try:
                        os.unlink(temp_bat)
                    except:
                        pass
                threading.Thread(target=cleanup, daemon=True).start()
                
            except Exception as e:
                # Fallback: just open terminal in the directory
                try:
                    subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', f'cd /d "{exe_dir}"'])
                    QMessageBox.information(
                        self, "Terminal Opened", 
                        f"Terminal opened in:\n{exe_dir}\n\nPlease run manually:\n{cmd_str}"
                    )
                except Exception as e2:
                    QMessageBox.critical(
                        self, "Launch Failed", 
                        f"Could not launch terminal.\n\nError: {str(e2)}\n\nCommand to run manually:\n{cmd_str}"
                    )
            
        elif sys.platform == "darwin":  # macOS
            # Use Terminal.app
            script = f'tell application "Terminal" to do script "cd \\"{exe_dir}\\" && {cmd_str}"'
            subprocess.Popen(["osascript", "-e", script])
            
        else:  # Linux and other Unix-like systems
            # Try common terminal emulators
            terminals = ["gnome-terminal", "konsole", "xterm", "x-terminal-emulator"]
            
            for terminal in terminals:
                try:
                    if terminal == "gnome-terminal":
                        subprocess.Popen([
                            terminal, "--", "bash", "-c", 
                            f"cd '{exe_dir}' && {cmd_str}; exec bash"
                        ])
                    elif terminal == "konsole":
                        subprocess.Popen([
                            terminal, "-e", "bash", "-c",
                            f"cd '{exe_dir}' && {cmd_str}; exec bash"
                        ])
                    else:
                        subprocess.Popen([
                            terminal, "-e", "bash", "-c",
                            f"cd '{exe_dir}' && {cmd_str}; exec bash"
                        ])
                    break
                except FileNotFoundError:
                    continue
            else:
                # Fallback: launch without terminal (background process)
                subprocess.Popen(cmd, cwd=exe_dir)
                QMessageBox.information(
                    self, "Info", 
                    "No suitable terminal found. Process launched in background."
                )
                
    def save_model_config(self):
        """Save current model configuration"""
        if not self.current_model:
            return
            
        config = self.parameter_widget.get_config()
        config.model_path = self.current_model['path']
        
        self.model_configs[self.current_model['path']] = config
        self.save_settings()
        self.statusBar().showMessage("Configuration saved")
        
    def load_model_config(self):
        """Load model configuration"""
        if not self.current_model:
            return
            
        model_path = self.current_model['path']
        if model_path in self.model_configs:
            config = self.model_configs[model_path]
            self.parameter_widget.set_config(config)
            self.gpu_layers_slider.setValue(config.n_gpu_layers)
            self.statusBar().showMessage("Configuration loaded")
            
    def load_model_config_if_exists(self, model_path: str):
        """Load model config if it exists"""
        if model_path in self.model_configs:
            config = self.model_configs[model_path]
            self.parameter_widget.set_config(config)
            self.gpu_layers_slider.setValue(config.n_gpu_layers)
            
    def get_settings_file(self) -> str:
        """Get settings file path"""
        return os.path.join(os.path.dirname(__file__), "launcher_settings.json")
        
    def save_settings(self):
        """Save application settings"""
        settings = {
            'global_config': asdict(self.global_config),
            'model_configs': {path: asdict(config) for path, config in self.model_configs.items()}
        }
        
        try:
            with open(self.get_settings_file(), 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    def load_settings(self):
        """Load application settings"""
        settings_file = self.get_settings_file()
        if not os.path.exists(settings_file):
            return
            
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                
            # Load global config
            if 'global_config' in settings:
                global_data = settings['global_config']
                self.global_config = GlobalConfig(**global_data)
                
                # Apply loaded settings
                self.models_dir_edit.setText(self.global_config.models_directory)
                self.executable_edit.setText(self.global_config.executable_path)
                
            # Load model configs
            if 'model_configs' in settings:
                self.model_configs = {
                    path: ModelConfig(**config_data) 
                    for path, config_data in settings['model_configs'].items()
                }
                
        except Exception as e:
            print(f"Error loading settings: {e}")
            
    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Llama.cpp Enhanced Launcher")
    app.setApplicationVersion("1.0")
    
    # Set application icon if available
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
        
    launcher = LlamaLauncher()
    launcher.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()