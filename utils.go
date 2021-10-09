package main

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	d "./db"
)

// ExtensionByContent ... Returns extension of file by detecting its MIME type, `.none` returned if no MIME found
func ExtensionByContent(content []byte) string {
	contentType := http.DetectContentType([]byte(content))
	splitted := strings.Split(contentType, "; ")[0]
	extenstions := map[string]string{"text/xml": ".xml", "text/html": ".html", "application/pdf": ".pdf", "text/plain": ".txt", "application/msword": ".doc"}
	if extenstions[splitted] == "" {
		//log.Println("[extensionByContent] No extension for " + splitted)
		return ".none"
	}
	return extenstions[splitted]
}

// EscapeURL ... Makes possible to create files with URL name by replacing system chars with %<char code>
func EscapeURL(url string) string {
	badChars := []string{"/", "\\", ":", "?"}
	escapedURL := ""

	for _, urlChar := range url {
		for i, badChar := range badChars {
			if string(urlChar) == badChar {
				conv := strconv.Itoa(int(urlChar))
				escapedURL += "%" + conv
				break
			} else if i == len(badChars)-1 {
				escapedURL += string(urlChar)
			}
		}
	}
	return escapedURL
}

// FilenameFromURL ... returns filename from URL
func FilenameFromURL(url string) string {
	items := strings.Split(url, "/")
	return items[len(items)-1]
}

func randString(len int) string {
	bytes := make([]byte, len)
	for i := 0; i < len; i++ {
		bytes[i] = byte(65 + rand.Intn(25)) //A=65 and Z = 65+25
	}
	return string(bytes)
}

func randomOption(options []string) string {
	rand.Seed(time.Now().Unix())
	randNum := rand.Int() % len(options)
	return options[randNum]
}

// CreateDir ... Create directory if not exists
func CreateDir(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.Mkdir(path, os.ModeDir)
		if err != nil {
			return fmt.Errorf("[createDir] error: %v", err)
		}
	}
	return nil
}

// CreateDirs ... Creates directories in chosen directory from array of strings
func CreateDirs(path string, dirs []string) error {
	var err error
	for _, dir := range dirs {
		err = CreateDir(path + "/" + dir)
	}
	return err
}

func getCompanyIndustry(c d.Companies) string {
	if c.IndustryGroups != "" {
		return c.IndustryGroups
	}
	return c.Industry
}

func logToFile(location string) *log.Logger {
	f, err := os.Create(location)
	if err != nil {
		panic(err)
	}

	logger := log.New(f, "", log.LstdFlags)
	logger.Println("Log started\n-------------------------------")
	return logger
}

// IsExtensionExist ...Return `true` if there is extension in allowed or false if there is no
// returns `true` if length of `allowedExtensions` == 0
func IsExtensionExist(allowedExtensions []string, extension string) bool {
	if len(allowedExtensions) == 0 {
		return true
	}
	for _, ext := range allowedExtensions {
		if ext == extension {
			return true
		}
	}
	return false
}
