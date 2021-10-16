package db

import (
	"fmt"
	"log"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/sqlite"
)

// Database ... Contains methods to work with data
type Database struct {
	*gorm.DB
	//busyCrawlIDs  []int
	//busyCollyIDs  []int
	//busyGoogleIDs []int
}

// OpenInitialize ... Open connection to DB and Initializes custom structure. Connection needs to be closed
func (db *Database) OpenInitialize(path string) {
	gdb, err := gorm.Open("sqlite3", path)
	if err != nil {
		log.Fatalln("failed to connect database: ", err)
	}
	//defer gdb.Close()
	gdb.Exec("PRAGMA foreign_keys = ON;")
	gdb.SingularTable(true)
	gdb.LogMode(false)
	gdb.AutoMigrate(&Economics{}, &Businesses{}, &IndustryGroups{}, &Industries{}, &Companies{})
	db.DB = gdb

	// Exclude 0 indexes, since they always have empty values in SQLite
	//db.busyCollyIDs = []int{0}
	//db.busyCrawlIDs = []int{0}
	//db.busyGoogleIDs = []int{0}
}

// PrintInfo ... Prints basic info about items in database
func (db *Database) PrintInfo() {
	industries := []Industries{}
	db.DB.Find(&industries)
	fmt.Println("Industries in DB: ", len(industries))

	companies := []Companies{}
	db.Find(&companies)
	fmt.Println("Companies in DB: ", len(companies))

	fmt.Println("Not crawled:")
	common := db.GetCommon()
	fmt.Println(" Common crawl: ", len(common))
	google := db.GetGoogle()
	fmt.Println(" Google filter: ", len(google))
	colly := db.GetColly()
	fmt.Println(" Colly crawler: ", len(colly))
}

/*
func (db *Database) GetCollyURL() (string, error) {
	company := Companies{}
	db.Where("id NOT IN (?)", db.busyCollyIDs).Where(&Companies{IsCollyCrawled: false}).Find(&company)
	if company.ID == 0 {
		return "", errors.New("[GetCollyURL] no URLs found")
	}
	db.busyCollyIDs = append(db.busyCollyIDs, company.ID)
	fmt.Println("busy colly: ", db.busyCollyIDs)
	return company.URL, nil
}
*/

func (db *Database) GetCommon() []Companies {
	companies := []Companies{}
	db.Where("is_common_crawled != 1").Find(&companies)
	return companies
}

func (db *Database) CommonFinished(url string) {
	company := Companies{URL: url}
	db.Model(&company).Where("url = ?", url).Update("is_common_crawled", true)
	company.IsCommonCrawled = true
	db.Save(&company)
}

func (db *Database) GetGoogle() []Companies {
	companies := []Companies{}
	db.Where("is_google_crawled != 1").Find(&companies)
	return companies
}

func (db *Database) GoogleFinished(url string) {
	company := Companies{URL: url}
	db.Model(&company).Where("url = ?", url).Update("is_google_crawled", true)
	company.IsGoogleCrawled = true
	db.Save(&company)
}

func (db *Database) GetColly() []Companies {
	companies := []Companies{}
	db.Where("is_colly_crawled != 1").Find(&companies)
	return companies
}

func (db *Database) CollyFinished(url string) {
	company := Companies{URL: url}
	db.Model(&company).Where("url = ?", url).Update("is_colly_crawled", true)
	company.IsCollyCrawled = true
	db.Save(&company)
}

func (db *Database) fillToDebug() {
	testIndustr := []Industries{
		Industries{Industry: "Internet Services"},
		Industries{Industry: "Software Developement"},
		Industries{Industry: "Education"},
		Industries{Industry: "Retail"},
	}

	for _, ind := range testIndustr {
		db.Create(&ind)
	}

	testCompanies := []Companies{
		Companies{URL: "https://domru.ru", Industry: "Internet Services"},
		Companies{URL: "https://innopolis.ru", Industry: "Education"},
		Companies{URL: "https://tattelecom.ru/", Industry: "Internet Services"},
		Companies{URL: "https://www.wikipedia.org", Industry: "Education"},
		Companies{URL: "https://kai.ru", Industry: "Education"},
		Companies{URL: "https://www.acronis.com", Industry: "Software Developement"},
		Companies{URL: "https://www.kaspersky.ru", Industry: "Software Developement"},
		Companies{URL: "http://pivoman-kazan.ru", Industry: "Retail"},
	}
	for _, comp := range testCompanies {
		isPossible := db.NewRecord(&comp)
		if !isPossible {
			log.Println("Cannot insert value: ", &comp)
		}
		db.Create(&comp)
	}
}

// GetIndustriesFolders ... Returns folders of industry categories, folders of which should be created
func (db *Database) GetIndustriesFolders() []string {
	folders := []string{}
	companies := []Companies{}

	// Get all companies that have Industry Group and make a set of these group names
	db.Where("industry_groups != 0").Find(&companies)
	industryGroupSet := map[string]struct{}{}
	for _, c := range companies {
		industryGroupSet[c.IndustryGroups] = struct{}{}
	}

	// Get all companies that have Industry and do not have Industry group, then make a set of these industry names
	db.Where("industry != 0 and industry_groups is NULL").Find(&companies)
	industrySet := map[string]struct{}{}
	for _, c := range companies {
		industrySet[c.Industry] = struct{}{}
	}

	// Join the 2 sets
	for ig := range industryGroupSet {
		folders = append(folders, ig)
	}
	for i := range industrySet {
		folders = append(folders, i)
	}
	return folders
}

func main() {
	//os.Remove("test.db")
	db := Database{}
	db.OpenInitialize("test.db")
	db.fillToDebug()

	db.PrintInfo()
	db.Close()
}
