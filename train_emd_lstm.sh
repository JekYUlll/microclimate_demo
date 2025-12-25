python scripts/emd_lstm.py \
  --data data/AntAWS/3_hourly/Taishan_3h.csv \
  --device cuda \
  --max-imfs 2 \
  --decompose-cache log/taishan_emd_cache.npz \
  --log-path log/emd_lstm_taishan.log
