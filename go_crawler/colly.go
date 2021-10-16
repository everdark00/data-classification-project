package main

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"

	cly "github.com/gocolly/colly"
)

// CollyResultChan ... result of work of `CrawlSite` function
type CollyResultChan struct {
	URL    string
	Error  error
	Loaded uint
	Done   bool
}

// CollyConfig ... Holds configuration parameters for Colly crawler
type CollyConfig struct {
	ResChanel     chan CollyResultChan
	MaxFileSize   int
	MaxHTMLLoad   uint
	WorkMinutes   int
	MaxAmount     int
	Extensions    []string
	RandomizeName bool
}

// CrawlSite ... Crawl choosen URL and saves found files
func CrawlSite(urlSite string, saveto string, config CollyConfig) {
	defer func() {
		if r := recover(); r != nil {
		}
	}()

	url := "https://" + urlSite
	downloaded := 0
	var loadedSize uint
	maxLoadSize := config.MaxHTMLLoad * 1024
	waitTime := time.Minute * time.Duration(config.WorkMinutes)
	c := cly.NewCollector()
	c.AllowedDomains = []string{"www." + urlSite, "sso." + urlSite, urlSite, "s3.amazonaws.com"}
	c.WithTransport(&http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		Dial: (&net.Dialer{
			Timeout:   180 * time.Second,
			KeepAlive: 0,
		}).Dial,
		TLSHandshakeTimeout:   60 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: 60 * time.Second,
	})

	// Reduce maximum response body size to 1M
	size := int(1024 * 1024 * config.MaxFileSize)
	c.MaxBodySize = size

	c.OnHTML("a[href]", func(e *cly.HTMLElement) {
		link := e.Attr("href")
		c.Visit(e.Request.AbsoluteURL(link))
	})

	c.OnRequest(func(r *cly.Request) {
		r.Headers.Set("User-Agent", randomOption(userAgents))
		//fmt.Println("[Visiting]", r.URL.String())
	})
	c.OnError(func(_ *cly.Response, err error) {
		config.ResChanel <- CollyResultChan{URL: url, Error: err}
	})

	start := time.Now()
	c.OnResponse(func(r *cly.Response) {
		elapsed := time.Since(start)
		ext := ExtensionByContent(r.Body)
		// If colly worked more than set
		if (elapsed > waitTime) || (downloaded >= config.MaxAmount) {
			config.ResChanel <- CollyResultChan{URL: urlSite, Done: true, Loaded: loadedSize}
			panic("Exit")

		} else if ext == ".none" {
			return
		} else if loadedSize > maxLoadSize && !IsExtensionExist(config.Extensions, ext) {
			return
		}
		filename := EscapeURL(r.Request.URL.EscapedPath())
		// Some root pages are nameless and will not display at filesystem
		if filename == "" {
			filename = "index"
		}
		if config.RandomizeName {
			r.Save(saveto + "/" + filename + randString(6) + ext)
		} else {
			r.Save(saveto + "/" + filename + ext)
		}

		loadedSize += uint(len(r.Body) / 1024)
		downloaded++
	})

	c.Visit(url)

	// If it crawled less than 25 Kb - try again, but with `www.` domain
	if loadedSize < 1024*25 {
		url = "https://www." + urlSite
		c.Visit(url)
	}

	// Also try with HTTP, because some sites do not redirect
	if loadedSize < 1024*25 {
		url = "http://" + urlSite
		c.Visit(url)
	}
	if loadedSize < 1024*25 {
		url = "http://www." + urlSite
		c.Visit(url)
	}

	config.ResChanel <- CollyResultChan{URL: urlSite, Done: true, Loaded: loadedSize}
}
