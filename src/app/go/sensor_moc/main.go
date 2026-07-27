package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

type WeatherData struct {
	Station     string    `json:"station"`
	RecordTime  time.Time `json:"record_time"`
	Temperature float64   `json:"temperature"`
	Humidity    int       `json:"humidity"`
	WindDir     int       `json:"wind_dir"`
	WindSpeed   float64   `json:"wind_speed"`
}

func main() {
	dsn := os.Getenv("ANTARCTIC_DB_DSN")
	if dsn == "" {
		log.Fatal("ANTARCTIC_DB_DSN is required")
	}
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatalf("数据库连接失败: %v", err)
	}
	defer db.Close()

	// 查询历史数据
	rows, err := db.Query(`
		SELECT station, record_time, temperature, humidity, wind_dir, wind_speed
		FROM weather_data
		ORDER BY record_time ASC
	`)
	if err != nil {
		log.Fatalf("查询失败: %v", err)
	}
	defer rows.Close()

	client := &http.Client{}
	url := "http://localhost:8000/ingest"

	for rows.Next() {
		var d WeatherData
		err := rows.Scan(&d.Station, &d.RecordTime, &d.Temperature, &d.Humidity, &d.WindDir, &d.WindSpeed)
		if err != nil {
			log.Printf("读取行失败: %v", err)
			continue
		}

		body, _ := json.Marshal(d)

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
		if err != nil {
			log.Printf("构建请求失败: %v", err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			log.Printf("请求失败: %v", err)
			continue
		}
		resp.Body.Close()

		fmt.Printf("已发送数据: %+v\n", d)

		// 模拟传感器间隔
		time.Sleep(200 * time.Millisecond)
	}
}
