package main

import (
	"fmt"
	"net/url"
	"path"
	"sync"
	"time"

	cc "github.com/karust/gocommoncrawl"

	d "./db"
	"github.com/BurntSushi/toml"
)

// Miner ... Holds reference of database and does grouping of methods
type Miner struct {
	db              d.Database
	industryFolders []string
}

// CommonCrawl ... Crawler which uses Common Crawl web archive to get HTML pages and other data
func (m Miner) CommonCrawl(config commonConfig, wg *sync.WaitGroup) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(config.Path + "/common_log.txt")
	resChan := make(chan cc.Result)
	companies := m.db.GetCommon()
	workers := 0
	var innerWg sync.WaitGroup
	innerWg.Add(len(companies) + 1)

	// Track progress from goroutines via channel
	go func() {
		done := 0
		for r := range resChan {
			if r.Error != nil {
				logger.Printf("[CommonCrawl] Error occured: %v\n", r.Error)
			} else if r.Done {
				m.db.CommonFinished(r.URL)
				logger.Printf("Common done: %v\n", r.URL)
				done++
				workers--
				innerWg.Done()
			}

			// Debug output
			if config.Debug && r.Error != nil {
				fmt.Printf("Error occured: %v\n", r.Error)
			} else if config.Debug && r.Progress > 0 {
				fmt.Printf("Progress %v: %v/%v\n", r.URL, r.Progress, r.Total)
			} else if config.Debug && r.Done {
				fmt.Printf("Commo done: %v\n", r.URL)
			}

			// If amount of `Dones` equal to amount of companies, then exit loop
			if done == len(companies) {
				break
			}
		}
		innerWg.Done()
	}()

	for _, c := range companies {
		for workers >= config.Workers {
			time.Sleep(time.Second * 1)
		}

		saveFolder := path.Join(config.Path, getCompanyIndustry(c), url.PathEscape(c.URL))

		// Do not overload Index API server
		waitTime := time.Second * time.Duration(config.SearchInterval)
		start := time.Now()

		// Make config for parser
		commonConfig := cc.Config{ResultChan: resChan, Timeout: config.Timeout, CrawlDB: config.CrawlDB,
			WaitMS: config.WaitTime, Extensions: config.Extensions, MaxAmount: config.MaxAmount}

		go cc.FetchURLData(c.URL, saveFolder, commonConfig)
		workers++

		// Wait time before proceed cycle
		elapsed := time.Since(start)
		if elapsed < waitTime {
			time.Sleep(waitTime - elapsed)
		}
	}
	innerWg.Wait()
}

// GoogleCrawl ... Uses google search filters to find documents
func (m Miner) GoogleCrawl(config googleConfig, wg *sync.WaitGroup) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(config.Path + "/google_log.txt")
	resChan := make(chan GoogleResultChan)
	companies := m.db.GetGoogle()
	workers := 0
	var innerWg sync.WaitGroup
	innerWg.Add(len(companies) + 1)

	// Track progress from goroutines via channel
	go func() {
		done := 0
		for r := range resChan {
			if r.Error != nil {
				logger.Printf("Error occured [%v]: %v\n", r.URL, r.Error)
				done++
				workers--
				innerWg.Done()
			} else if r.Done {
				m.db.GoogleFinished(r.URL)
				logger.Printf("Google done: %v\n", r.URL)
				done++
				workers--
				innerWg.Done()
			} else if r.Warning != nil {
				logger.Printf("Warning [%v]: %v\n", r.URL, r.Warning)
			}

			// Debug output
			if config.Debug && r.Error != nil {
				fmt.Printf("Error occured [%v]: %v\n", r.URL, r.Error)
			} else if config.Debug && r.Progress > 0 {
				fmt.Printf("Progress %v: %v/%v\n", r.URL, r.Progress, r.Total)
			} else if config.Debug && r.Done {
				fmt.Printf("Google done: %v\n", r.URL)
			} else if config.Debug && r.Warning != nil {
				fmt.Printf("Warning [%v]: %v\n", r.URL, r.Warning)
			}

			// If amount of `Dones` equal to amount of companies, then exit loop
			if done == len(companies) {
				break
			}
		}
		innerWg.Done()
	}()

	for _, c := range companies {
		for workers >= config.Workers {
			time.Sleep(time.Second * 1)
		}

		saveFolder := path.Join(config.Path, getCompanyIndustry(c), url.PathEscape(c.URL))
		err := CreateDir(saveFolder)
		if err != nil && config.Debug {
			fmt.Println("[GoogleCrawl] error: ", err)
		}

		// Google search queries should not be too ofter, therefore launch goroutine with intervals
		waitTime := time.Second * time.Duration(config.SearchInterval)
		start := time.Now()

		go FetchURLFiles(c.URL, config.Extension, saveFolder, config.MaxFileSize, resChan)
		workers++

		// Wait time before next cycle
		elapsed := time.Since(start)
		if elapsed < waitTime {
			time.Sleep(waitTime - elapsed)
		}
	}
	innerWg.Wait()
}

// CollyCrawl ... Crawls each website by visiting links on them. Saves found PDF and HTML documents
func (m Miner) CollyCrawl(config collyConfig, wg *sync.WaitGroup) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(config.Path + "/colly_log.txt")
	resChan := make(chan CollyResultChan)
	companies := m.db.GetColly()
	workers := 0
	var innerWg sync.WaitGroup
	innerWg.Add(len(companies) + 1)

	// Track progress from goroutines via channel
	go func() {
		done := 0
		for r := range resChan {
			if r.Error != nil {
				logger.Printf("[CollyCrawl] Error occured: %v\n", r.Error)
			} else if r.Done && r.Loaded > 0 {
				// Save state in database
				m.db.CollyFinished(r.URL)
				logger.Printf("Colly done: %v\n", r.URL)
				done++
				workers--
				innerWg.Done()
			} else if r.Done && r.Loaded == 0 {
				logger.Printf("Colly failed: %v\n", r.URL)
				done++
				workers--
				innerWg.Done()
			}

			// Debug output
			if config.Debug && r.Error != nil {
				fmt.Printf("[CollyCrawl] Error occured: %v\n", r.Error)
			} else if config.Debug && r.Done && r.Loaded > 0 {
				fmt.Printf("Colly done: %v\n", r.URL)
			} else if config.Debug && r.Done && r.Loaded == 0 {
				fmt.Printf("Colly failed: %v\n", r.URL)
			}

			// If amount of `Dones` equal to amount of companies, then exit loop
			if done == len(companies) {
				break
			}
		}
		innerWg.Done()
	}()

	// Launch goroutine with crawler for each site
	for _, c := range companies {
		for workers >= config.Workers {
			time.Sleep(time.Second * 1)
		}
		saveFolder := path.Join(config.Path, getCompanyIndustry(c), url.PathEscape(c.URL))
		err := CreateDir(saveFolder)
		if err != nil {
			panic(err)
		}

		// Make configuration for crawler
		collyConfig := CollyConfig{ResChanel: resChan, MaxAmount: config.MaxAmount, Extensions: config.Extensions,
			MaxFileSize: config.MaxFileSize, MaxHTMLLoad: config.MaxHTMLLoad, WorkMinutes: config.WorkMinutes, RandomizeName: config.RandomName}

		go CrawlSite(c.URL, saveFolder, collyConfig)
		workers++
	}

	innerWg.Wait()
}

func main() {
	// Try to load configuration file, if error then meaningless to proceed
	var config Config
	if _, err := toml.DecodeFile("config.toml", &config); err != nil {
		fmt.Println("Config load error:")
		panic(err)
	}

	// Initialize miner and database
	miner := Miner{}
	miner.db = d.Database{}
	miner.db.OpenInitialize(config.General.Database)
	miner.db.PrintInfo()
	defer miner.db.Close()

	// Get insustry folders in which data will be saved in categorized way
	miner.industryFolders = miner.db.GetIndustriesFolders()

	var wg sync.WaitGroup
	// 1. Use CommonCrawl to retrive indexed HTML pages of given site
	if config.Common.Use {
		wg.Add(1)
		go miner.CommonCrawl(config.Common, &wg)
	}

	// 2. Use Google search with to find cached files
	if config.Google.Use {
		wg.Add(1)
		go miner.GoogleCrawl(config.Google, &wg)
	}

	// 3. Crawl site with GoColly to find unindexed documents
	if config.Colly.Use {
		wg.Add(1)
		go miner.CollyCrawl(config.Colly, &wg)
	}
	wg.Wait()
}
