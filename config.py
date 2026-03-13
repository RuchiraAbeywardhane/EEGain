from dataclasses import dataclass, field


@dataclass
class TransformConfig:
    sampling_r: int = 128
    window: int = 4
    overlap: int = 0

@dataclass
class MAHNOBConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5   # inclusive
    n_classes: int = 2

@dataclass
class DEAPConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5   # inclusive
    n_classes: int = 2
    
@dataclass
class AMIGOSConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5  # inclusive
    n_classes: int = 2

@dataclass
class DREAMERConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 3 # inclusive
    n_classes: int = 2
    
@dataclass
class SeedIVConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5  # inclusive
    n_classes: int = 4

@dataclass
class SeedConfig:
    data_path: str = ""
    label_type: str = "V"
    split_type: str = "LOSO"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5  # inclusive
    n_classes: int = 3

@dataclass
class EmognitionConfig:
    data_path: str = ""
    label_type: str = "V"          # unused, kept for API compatibility
    split_type: str = "LOSO_Fixed"
    class_names: list[str] = field(default_factory=lambda: ["enthusiasm", "neutral", "sadness", "fear"])
    ground_truth_threshold: float = 0.0   # unused
    n_classes: int = 4
    use_baseline_reduction: bool = True
    sampling_r: int = 256

@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4                  # lowered from 1e-3: avoids sharp, non-generalisable minima
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    num_epochs: int = 100
    early_stopping_patience: int = 15
    log_dir: str = "logs/"
    overal_log_file: str = "logs.txt"


@dataclass
class EEGNetConfig:
    num_classes: int = 2
    samples: int = 512
    dropout_rate: float = 0.5
    channels: int = 32

@dataclass
class TSceptionConfig:
    num_classes: int = 4
    sampling_r: int = 256
    num_t: int = 9
    num_s: int = 6
    hidden: int = 32
    dropout_rate: float = 0.5        # raised from 0.3: train/val gap was 0.97 nats (severe overfit)

@dataclass
class  DeepConvNetConfig:
    channels: int = 32
    num_classes: int = 2
    dropout_rate: int = 0.5
    
@dataclass
class ShallowConvNetConfig:
    channels: int = 32
    num_classes: int = 2
    dropout_rate: int = 0.5