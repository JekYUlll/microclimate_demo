package model

import "time"

type SnowDistribution struct {
	RawData    []byte    `json:"distribution"`
	RecordTime time.Time `json:"record_time"`
}

type FullData struct {
	// TODO: 风吹雪
	// Station     string    `json:"station"`
	RecordTime  time.Time `json:"record_time"`
	Temperature float64   `json:"temperature"`
	Humidity    int       `json:"humidity"`
	WindDir     int       `json:"wind_dir"`
	WindSpeed   float64   `json:"wind_speed"`
}
