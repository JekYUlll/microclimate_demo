python scripts/emd_lstm.py \
  --data data/AntAWS/3_hourly/Taishan_3h.csv \
  --device cuda \
  --max-imfs 2 \
  --epochs 15 \
  --log-path log/emd_lstm_taishan.log \
  --log-every 1 \
  --plot plots/emd_lstm_taishan.png