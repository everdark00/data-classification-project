package main

// Config ... Holds structure of TOML configuration file
type Config struct {
	General generalConfig
	Common  commonConfig
	Google  googleConfig
	Colly   collyConfig
}

type generalConfig struct {
	Database string
}

type commonConfig struct {
	Use        bool
	Path       string
	Debug      bool
	Extensions []string
	MaxAmount  int `toml:"max_amount"`
	//RandomName     bool `toml:"random_name"`
	//HashFilter     bool `toml:"hash_filter"`
	Timeout        int
	SearchInterval int    `toml:"search_interval"`
	CrawlDB        string `toml:"crawl_db"`
	WaitTime       int    `toml:"wait_time"`
	Workers        int
}

type googleConfig struct {
	Use            bool
	Path           string
	Debug          bool
	Extension      string
	SearchInterval int    `toml:"search_interval"`
	MaxFileSize    uint64 `toml:"max_file_size"`
	Workers        int
	//RandomName     bool `toml:"random_name"`
}

type collyConfig struct {
	Use         bool
	Path        string
	Debug       bool
	Extensions  []string
	MaxAmount   int  `toml:"max_amount"`
	MaxFileSize int  `toml:"max_file_size"`
	MaxHTMLLoad uint `toml:"max_html_load"`
	WorkMinutes int  `toml:"work_minutes"`
	Workers     int
	RandomName  bool `toml:"random_name"`
}
