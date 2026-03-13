@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  TSception on Emognition  (MUSE 4-channel, 4-class, LOSO_Fixed split)
REM  Change --data_path to wherever your Emognition JSON files live.
REM ─────────────────────────────────────────────────────────────────────────────

python run_cli.py ^
  --model_name=TSception ^
  --data_name=Emognition ^
  --data_path="C:\path\to\emognition\data" ^
  --data_config=EmognitionConfig ^
  --split_type=LOSO_Fixed ^
  --num_classes=4 ^
  --channels=4 ^
  --sampling_r=256 ^
  --window=4 ^
  --overlap=0 ^
  --num_t=9 ^
  --num_s=6 ^
  --hidden=32 ^
  --dropout_rate=0.3 ^
  --num_epochs=100 ^
  --early_stopping_patience=15 ^
  --batch_size=32 ^
  --lr=1e-4 ^
  --weight_decay=0.05 ^
  --label_smoothing=0.1 ^
  --train_val_split=0.8 ^
  --random_seed=2025 ^
  --log_dir=logs/ ^
  --overal_log_file=emognition_tsception.txt ^
  --log_predictions=False
