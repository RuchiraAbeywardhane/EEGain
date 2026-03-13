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
    lr: float = 1e-4
    weight_decay: float = 0.05        # increased: AdamW decouples it correctly now
    label_smoothing: float = 0.1      # increased: stronger regularisation
    num_epochs: int = 100             # high ceiling — early stopping will cut it
    early_stopping_patience: int = 15  # raised from 7: cross-subject val loss oscillates heavily
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
    num_classes: int = 4             # 4 emotions for Emognition (override to 2 for DEAP)
    sampling_r: int = 256            # 256 Hz for Emognition (override to 128 for DEAP)
    num_t: int = 9                   # temporal filters
    num_s: int = 6                   # spatial filters
    hidden: int = 32                 # FC hidden size (num_s -> hidden -> num_classes)
    dropout_rate: float = 0.3

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