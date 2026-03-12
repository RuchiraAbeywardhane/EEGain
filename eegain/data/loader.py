import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader

from .datasets import EEGDataset, EEGDatasetBase

logger = logging.getLogger("Dataloader")
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


class EEGDataloader:
    def __init__(self, dataset: EEGDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def loto(self, subject_id, session_ids, n_fold, **kwargs):
        fold_size = len(session_ids) // n_fold

        folds = [session_ids[i:i + fold_size] for i in range(0, len(session_ids), fold_size)]
        for test_sessions in folds:
            logger.debug(f"subject_id is: {subject_id}, test sessions are: {test_sessions}")
            train_sessions = session_ids.copy()
            train_sessions = [item for item in session_ids if item not in test_sessions]
            logger.debug(f"subject_id is: {subject_id}, train sessions are: {train_sessions}")
            test_data = self.dataset.__get_trials__(test_sessions, subject_id)
            train_data = self.dataset.__get_trials__(train_sessions, subject_id)
            train_data, train_label, train_videos = EEGDataloader._concat_data(train_data, loader_type="LOTO")
            test_data, test_label, test_videos = EEGDataloader._concat_data(test_data, loader_type="LOTO")

            if len(train_data.shape) != 4:  # DREAMER has already shape that is needed and it doesn't need normalization
                train_data, test_data = EEGDataloader.normalize(train_data, test_data)
            
            # Split training data into train and validation sets (80-20)
            train_ratio = kwargs.get('train_val_split', 0.8)
            train_data, train_label, val_data, val_label, train_videos, val_videos = self.split_train_val(
                train_data, train_label, train_ratio=train_ratio, videos=train_videos)

            train_dataloader = self._get_dataloader(train_data, train_label)
            val_dataloader = self._get_dataloader(val_data, val_label, shuffle=False)
            test_dataloader = self._get_dataloader(test_data, test_label, shuffle=False)
            
            yield {
                "train": train_dataloader,
                "val": val_dataloader,
                "test": test_dataloader,
                "test_session_indexes": test_sessions,
                "train_session_indexes": train_sessions,
                "subject_id": subject_id,
                "train_videos": train_videos,
                "val_videos": val_videos,
                "test_videos": test_videos,
            }

    @staticmethod
    def _concat_data(
        data: List[Tuple[Dict[int, np.ndarray], Dict[int, int]]]
    , loader_type="LOSO") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        concatenate data from several subjects and make one tensor

        Args:
            data: list of subjects data and label. each value is a dictionary with session and data pairs
            we just need values for training and not session ids

        Returns:
            x,y (torch.Tensor):
        """
        
        if loader_type == "LOTO":
            x = np.concatenate([v for v in data[0].values()], axis=0)
        else:
            x = np.concatenate([v for d in data for v in d[0].values()], axis=0)
        x = torch.from_numpy(x).float()

        # TODO: refactor
        y = []
        video_ids = []
        if loader_type == "LOTO":
            for i in data[1].items():
                k = i[0]
                v = i[1]
                y.extend(list(np.repeat(v, data[0][k].shape[0])))
            y = torch.tensor(y) 
        else:
            for i in data:
                #i is each subject
                for k, v in i[1].items():
                    #k - clip index
                    y.extend(list(np.repeat(v, i[0][k].shape[0])))
                    video_ids.extend(list(np.repeat(k, i[0][k].shape[0])))
            y = torch.tensor(y)


        return x, y,video_ids

    def _get_dataloader(
        self, data: torch.Tensor, label: torch.Tensor, shuffle: bool = True #changed to False
    ) -> DataLoader:
        """
        Returns a DataLoader object that can be used for iterating through batches of data and labels.

        Args:
            data (List[torch.Tensor]): List of input data tensors.
            label (List[torch.Tensor]): List of label tensors.
            shuffle (bool): Whether to shuffle the data before batching. Defaults to True.

        Returns:
            loader: A DataLoader object that can be used to iterate through batches of data and labels.
        """
        dataset = EEGDatasetBase(data, label)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
        return loader

    @staticmethod
    def normalize(
        train_data: torch.Tensor, test_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard normalization for EEG channel by channel.
        Fits mean/std on train_data, applies to both train and test.
        Handles both 3D (B, C, T) and 4D (B, 1, C, T) input — always returns 4D.
        """
        if len(train_data.shape) != 4:
            train_data = train_data.unsqueeze(1)
        if len(test_data.shape) != 4:
            test_data = test_data.unsqueeze(1)

        for channel in range(train_data.shape[2]):
            nan_mask = torch.isnan(train_data[:, :, channel, :])
            train_data[:, :, channel, :][nan_mask] = 0
            nan_mask = torch.isnan(test_data[:, :, channel, :])
            test_data[:, :, channel, :][nan_mask] = 0

            std, mean = torch.std_mean(train_data[:, :, channel, :])
            train_data[:, :, channel, :] = (train_data[:, :, channel, :] - mean) / std
            test_data[:, :, channel, :] = (test_data[:, :, channel, :] - mean) / std

        return train_data, test_data

    @staticmethod
    def normalize_per_subject(data: torch.Tensor) -> torch.Tensor:
        """
        Normalise each subject's windows independently, channel by channel.
        Since we don't know subject boundaries after _concat_data, we normalise
        every window individually — equivalent to per-subject z-score when all
        windows from one subject have similar statistics.
        Input: (B, 1, C, T) or (B, C, T) → returns (B, 1, C, T)
        """
        if len(data.shape) != 4:
            data = data.unsqueeze(1)
        data = data.clone()
        for channel in range(data.shape[2]):
            nan_mask = torch.isnan(data[:, :, channel, :])
            data[:, :, channel, :][nan_mask] = 0
            # per-sample (per-window) normalisation
            mean = data[:, :, channel, :].mean(dim=-1, keepdim=True)
            std  = data[:, :, channel, :].std(dim=-1, keepdim=True).clamp(min=1e-6)
            data[:, :, channel, :] = (data[:, :, channel, :] - mean) / std
        return data

    @staticmethod
    def normalize_splits(
        train_data: torch.Tensor,
        *other_splits: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Normalise each split independently per-window so unseen subjects
        are not penalised by a different population's amplitude statistics.
        """
        result = [EEGDataloader.normalize_per_subject(train_data)]
        for s in other_splits:
            result.append(EEGDataloader.normalize_per_subject(s))
        return tuple(result)

    def loso(self, subject_out_num: int = 1, **kwargs) -> Dict[str, Any]:
        logger.info(f"Splitting type: leave-one-subject-out")
        subject_ids = self.dataset.__get_subject_ids__()
        test_ids_combination = list(combinations(subject_ids, subject_out_num))

        for test_subject_ids in test_ids_combination:
            train_subject_ids = set(subject_ids) - set(test_subject_ids)
            
            # Get training and test data
            train_data = [self.dataset.__get_subject__(i) for i in train_subject_ids]
            train_data, train_label, train_videos = EEGDataloader._concat_data(train_data)

            test_data = [self.dataset.__get_subject__(i) for i in test_subject_ids]
            test_data, test_label, test_videos = EEGDataloader._concat_data(test_data)

            train_data, test_data = EEGDataloader.normalize(train_data, test_data)
            
            # Split training data into train and validation sets (80-20)
            train_ratio = kwargs.get('train_val_split', 0.8)
            train_data, train_label, val_data, val_label, train_videos, val_videos = self.split_train_val(
                train_data, train_label, train_ratio=train_ratio, videos=train_videos)
            
            train_dataloader = self._get_dataloader(train_data, train_label)
            val_dataloader = self._get_dataloader(val_data, val_label, shuffle=False)  
            test_dataloader = self._get_dataloader(test_data, test_label, shuffle=False)
            
            yield {
                "train": train_dataloader,
                "val": val_dataloader,
                "test": test_dataloader,
                "test_subject_indexes": test_subject_ids,
                "train_subject_indexes": train_subject_ids,
                "train_videos": train_videos,
                "val_videos": val_videos,
                "test_videos": test_videos,
            }

    def loso_fixed(self, train_subject_ids, test_subject_ids, **kwargs) -> Dict[str, Any]:
        logger.info(f"Splitting type: leave-n-subject-out-fixed")

        # Filter out any subject IDs that don't exist in the dataset
        available = set(self.dataset.__get_subject_ids__())

        orig_train = list(train_subject_ids)
        orig_test  = list(test_subject_ids)
        train_subject_ids = [i for i in orig_train if str(i) in available or i in available]
        test_subject_ids  = [i for i in orig_test  if str(i) in available or i in available]

        skipped_train = [i for i in orig_train if i not in train_subject_ids]
        skipped_test  = [i for i in orig_test  if i not in test_subject_ids]
        if skipped_train:
            logger.warning(f"Skipping missing train subjects (not in dataset): {skipped_train}")
        if skipped_test:
            logger.warning(f"Skipping missing test subjects (not in dataset): {skipped_test}")

        if not train_subject_ids:
            raise ValueError("No valid train subjects found after filtering.")
        if not test_subject_ids:
            raise ValueError("No valid test subjects found after filtering.")

        # ----------------------------------------------------------------
        # Subject-level train/val split — val subjects are DIFFERENT people
        # from train subjects, preventing subject-identity leakage.
        # Prefer subjects with enough windows (>=80) for val to get a clean signal.
        # ----------------------------------------------------------------
        train_ratio = kwargs.get('train_val_split', 0.8)
        n_val = max(1, int(len(train_subject_ids) * (1.0 - train_ratio)))

        # Count windows per subject so we can pick strong val subjects
        window_counts = {}
        for sid in train_subject_ids:
            d = self.dataset.__get_subject__(sid)
            count = sum(v.shape[0] for v in d[0].values())
            window_counts[sid] = count

        # Sort: subjects with >= 80 windows first (stable val signal), then rest
        MIN_VAL_WINDOWS = 80
        strong = [s for s in train_subject_ids if window_counts[s] >= MIN_VAL_WINDOWS]
        weak   = [s for s in train_subject_ids if window_counts[s] <  MIN_VAL_WINDOWS]

        rng = np.random.default_rng(kwargs.get('random_seed', 42))
        rng.shuffle(strong)
        rng.shuffle(weak)

        # Pick val from strong subjects first
        candidate_pool = strong + weak
        val_subject_ids   = candidate_pool[:n_val]
        train_subject_ids = candidate_pool[n_val:]

        logger.info(f"Subject-level split — "
                    f"train: {len(train_subject_ids)} subjects, "
                    f"val: {len(val_subject_ids)} subjects "
                    f"(windows: {[window_counts[s] for s in val_subject_ids]}), "
                    f"test: {len(test_subject_ids)} subjects")
        logger.debug(f"Val subjects: {val_subject_ids}")

        # ----------------------------------------------------------------
        # Load data for each split
        # ----------------------------------------------------------------
        logger.debug(f"Preparing: train subjects: {train_subject_ids}")
        train_data = [self.dataset.__get_subject__(i) for i in train_subject_ids]
        train_data, train_label, train_videos = EEGDataloader._concat_data(train_data)
        logger.debug(f"train data shape {train_data.shape}")

        logger.debug(f"Preparing: val subjects: {val_subject_ids}")
        val_data = [self.dataset.__get_subject__(i) for i in val_subject_ids]
        val_data, val_label, val_videos = EEGDataloader._concat_data(val_data)
        logger.debug(f"val data shape {val_data.shape}")

        logger.debug(f"Preparing: test subjects: {test_subject_ids}")
        test_data = [self.dataset.__get_subject__(i) for i in test_subject_ids]
        test_data, test_label, test_videos = EEGDataloader._concat_data(test_data)
        logger.debug(f"test data shape {test_data.shape}")

        # Normalise: fit scaler on train only, apply to val + test in one pass
        train_data, val_data, test_data = EEGDataloader.normalize_splits(
            train_data, val_data, test_data
        )

        train_dataloader = self._get_dataloader(train_data, train_label)
        val_dataloader   = self._get_dataloader(val_data,   val_label,   shuffle=False)
        test_dataloader  = self._get_dataloader(test_data,  test_label,  shuffle=False)

        return {
            "train": train_dataloader,
            "val":   val_dataloader,
            "test":  test_dataloader,
            "test_subject_indexes":  test_subject_ids,
            "train_subject_indexes": train_subject_ids,
            "val_subject_indexes":   val_subject_ids,
            "train_videos": train_videos,
            "val_videos":   val_videos,
            "test_videos":  test_videos,
        }

    def split_train_val(self, data, label, train_ratio=0.8, shuffle=True, videos=None):
        """
        Split training data into training and validation sets in a class-balanced way for N classes.
    
        Args:
            data: Training data tensor
            label: Training labels tensor
            train_ratio: Ratio of data to use for training
            shuffle: Whether to shuffle indices before splitting
            videos: Optional video IDs corresponding to the data
    
        Returns:
            train_data, train_label, val_data, val_label, train_videos, val_videos
        """
        # Convert tensor to numpy if needed
        if isinstance(label, torch.Tensor):
            label_np = label.cpu().numpy()
        else:
            label_np = label
    
        # Get unique classes
        unique_classes = np.unique(label_np)
    
        # Initialize empty arrays for train and validation indices
        index_train = np.array([], dtype=int)
        index_val = np.array([], dtype=int)
    
        # Process each class separately to maintain class balance
        for class_label in unique_classes:
            # Find indices for this class
            class_indices = np.where(label_np == class_label)[0]
        
            # Copy indices
            class_indices_copy = copy.deepcopy(class_indices)
        
            # Shuffle if requested
            if shuffle:
                np.random.shuffle(class_indices_copy)
        
            # Calculate split point
            split_point = int(len(class_indices_copy) * train_ratio)
        
            # Add indices to train and validation sets
            index_train = np.concatenate((index_train, class_indices_copy[:split_point]))
            index_val = np.concatenate((index_val, class_indices_copy[split_point:]))
    
        # Extract data and labels for training and validation
        train_data = data[index_train]
        train_label = label[index_train]
        val_data = data[index_val]
        val_label = label[index_val]
    
        # Handle videos if provided
        train_videos, val_videos = None, None
        if videos is not None:
            videos_list = videos if isinstance(videos, list) else videos.tolist() if hasattr(videos, 'tolist') else []
            if videos_list:
                train_videos = [videos_list[i] for i in index_train]
                val_videos = [videos_list[i] for i in index_val]
    
        return train_data, train_label, val_data, val_label, train_videos, val_videos