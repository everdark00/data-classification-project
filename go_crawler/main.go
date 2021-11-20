package main

import (
	"fmt"
	"log"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	cc "github.com/karust/gocommoncrawl"

	"github.com/BurntSushi/toml"
	d "gitlab.com/alekseik1/dataclassification-crawler/db"
)

// Miner ... Holds reference of database and does grouping of methods
type Miner struct {
	db              d.Database
	industryFolders []string
}

// CommonCrawl ... Crawler which uses Common Crawl web archive to get HTML pages and other data
func (m Miner) CommonCrawl(config commonConfig, wg *sync.WaitGroup, logPath string) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(logPath+"/common_log.txt", "[CommonCrawl] ")
	companies := m.db.GetCommon()
	// initialize channels to communicate with each crawler
	resChan := make(map[int]chan cc.Result, len(companies))
	workers := 0
	var innerWg sync.WaitGroup
	innerWg.Add(len(companies))

	// Track progress from goroutines via channel
	url_done := 0
	handleOne := func(resChannel chan cc.Result, num int) {
		logger.Printf("Starting waiting for events for channel #%v", num)
		for r := range resChannel {
			if r.Error != nil {
				logger.Printf("#%v Error occurred: %v\n", num, r.Error)
				// NOTE if GetPagesInfo error, an exit is required since NO MORE messages will come
				// see src codes for details
				if strings.Contains(r.Error.Error(), "GetPagesInfo") {
					logger.Printf("#%v Exiting after GetPagesInfo error", num)
					break
				}
			}
			if r.Done {
				m.db.CommonFinished(r.URL)
				logger.Printf("#%v URL is processed: %v\n", num, r.URL)
				break
			}
			if config.Debug && r.Progress > 0 {
				logger.Printf("[DEBUG] #%v Progress %v: %v/%v\n", num, r.URL, r.Progress, r.Total)
			}

		}
		// NOTE after implementing "reschedule broken URL" feature url_done should not increment here
		url_done += 1
		workers -= 1
		logger.Printf("#%v TOTAL done: %v / %v", num, url_done, len(companies))
		innerWg.Done()
	}
	for i, c := range companies {
		for workers >= config.Workers {
			time.Sleep(time.Second * 1)
		}
		logger.Printf("creating crawler for company #%v with url %v", i, c.URL)

		saveFolder := path.Join(config.Path, getCompanyIndustry(c), url.PathEscape(c.URL))
		if config.Debug {
			logger.Printf("[DEBUG] creating folder %v", saveFolder)
		}
		// Manually create folder since library does not handle permissions properly
		err := os.Mkdir(saveFolder, 0755)
		if err != nil {
			fmt.Printf("Error during folder creaion %v", err)
		}
		if config.Debug {
			logger.Printf("[DEBUG] (DONE) creating folder %v", saveFolder)
		}

		// Do not overload Index API server
		logger.Printf("Waiting %v seconds before starting new worker", config.SearchInterval)
		waitTime := time.Second * time.Duration(config.SearchInterval)
		start := time.Now()
		for {
			// Wait time before proceed cycle
			elapsed := time.Since(start)
			if elapsed < waitTime {
				time.Sleep(waitTime - elapsed)
			} else {
				break
			}
		}

		if config.Debug {
			logger.Printf("[DEBUG] creating result channel & running handling for URL #%v", i)
		}
		resChan[i] = make(chan cc.Result)
		go handleOne(resChan[i], i)
		if config.Debug {
			logger.Printf("[DEBUG] (DONE) creating result channel & running handling for URL #%v", i)
		}

		// Make config for parser
		commonConfig := cc.Config{ResultChan: resChan[i], Timeout: config.Timeout, CrawlDB: config.CrawlDB,
			WaitMS: config.WaitTime, Extensions: config.Extensions, MaxAmount: config.MaxAmount}

		go cc.FetchURLData(c.URL, saveFolder, commonConfig)
		workers++
		logger.Printf("(DONE) creating crawler for company #%v with url %v", i, c.URL)
	}

	innerWg.Wait()
	logger.Printf("All URLs are processed, exiting")
}

// GoogleCrawl ... Uses google search filters to find documents
func (m Miner) GoogleCrawl(config googleConfig, wg *sync.WaitGroup, logPath string) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(logPath+"/google_log.txt", "[GoogleCrawl] ")
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
				logger.Printf("Error occurred [%v]: %v\n", r.URL, r.Error)
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
			} else if config.Debug && r.Progress > 0 {
				fmt.Printf("Progress %v: %v/%v\n", r.URL, r.Progress, r.Total)
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

		go FetchURLFiles(c.URL, config.Extension, saveFolder, config.MaxFileSize, resChan, config.RandomSeed)
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
func (m Miner) CollyCrawl(config collyConfig, wg *sync.WaitGroup, logPath string) {
	defer wg.Done()

	// Create directories in which data from sites will be saved
	err := CreateDirs(config.Path, m.industryFolders)
	if err != nil {
		fmt.Printf("[CollyCrawl] Fatal error occured: %v\n", err)
		return
	}

	// Initialize variables
	logger := logToFile(logPath+"/colly_log.txt", "[CollyCrawl] ")
	companies := m.db.GetColly()
	workers := 0
	resChan := make(map[int]chan CollyResultChan, len(companies))
	var innerWg sync.WaitGroup
	wg_total := len(companies)
	logger.Printf("[CollyCrawl] adding %v to WaitGroup\n", wg_total)
	innerWg.Add(wg_total)

	// Track progress from goroutines via channel
	handleOne := func(resChan chan CollyResultChan, num int) {
		done := 0
		for r := range resChan {
			if r.Error != nil {
				logger.Printf("[CollyCrawl] #%v Error occurred: %v\n", num, r.Error)
				//if strings.Contains(r.Error.Error(), "context deadline exceeded") {
				//	logger.Printf("[CollyCrawl] #%v death error occurred, exiting worker", num)
				//	workers--
				//	innerWg.Done()
				//}
			} else if r.Done && r.Loaded > 0 {
				// Save state in database
				m.db.CollyFinished(r.URL)
				logger.Printf("[CollyCrawl] #%v Colly done: %v\n", num, r.URL)
				done++
				workers--
				logger.Printf("[CollyCrawl] wg.Done() after success\n")
				innerWg.Done()
			} else if r.Done && r.Loaded == 0 {
				logger.Printf("[CollyCrawl] #%v Colly failed: %v\n", num, r.URL)
				done++
				workers--
				logger.Printf("[CollyCrawl] wg.Done() after failure")
				innerWg.Done()
			}

			// If amount of `Dones` equal to amount of companies, then exit loop
			if done == len(companies) {
				break
			}
		}
		// logger.Printf("[CollyCrawl] wg.Done() after channel close")
		// innerWg.Done()
	}

	// Launch goroutine with crawler for each site
	for i, c := range companies {
		for workers >= config.Workers {
			time.Sleep(time.Second * 1)
		}
		saveFolder := path.Join(config.Path, getCompanyIndustry(c), url.PathEscape(c.URL))
		err := CreateDir(saveFolder)
		if err != nil {
			panic(err)
		}

		resChan[i] = make(chan CollyResultChan)
		// Make configuration for crawler
		collyConfig := CollyConfig{ResChanel: resChan[i], MaxAmount: config.MaxAmount, Extensions: config.Extensions,
			MaxFileSize: config.MaxFileSize, MaxHTMLLoad: config.MaxHTMLLoad, WorkMinutes: config.WorkMinutes, RandomizeName: config.RandomName, RandomSeed: config.RandomSeed}

		logger.Printf("[CollyCrawl] running handler for #%v", i)
		go handleOne(resChan[i], i)
		logger.Printf("[CollyCrawl] running crawler for #%v", i)
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
	defer func(db *d.Database) {
		err := db.Close()
		if err != nil {
			log.Panicln("Error when closing database")
		}
	}(&miner.db)

	// Get insustry folders in which data will be saved in categorized way
	miner.industryFolders = miner.db.GetIndustriesFolders()

	var wg sync.WaitGroup
	// 1. Use CommonCrawl to retrive indexed HTML pages of given site
	if config.Common.Use {
		wg.Add(1)
		go miner.CommonCrawl(config.Common, &wg, config.General.LogPath)
	}

	// 2. Use Google search with to find cached files
	if config.Google.Use {
		wg.Add(1)
		go miner.GoogleCrawl(config.Google, &wg, config.General.LogPath)
	}

	// 3. Crawl site with GoColly to find unindexed documents
	if config.Colly.Use {
		wg.Add(1)
		go miner.CollyCrawl(config.Colly, &wg, config.General.LogPath)
	}
	wg.Wait()
}
