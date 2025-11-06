conda activate darts
# pip install "u8darts[all]"
uvicorn app:app --reload --host 0.0.0.0 --port 8000

go run ./go/sensor_moc/main.go