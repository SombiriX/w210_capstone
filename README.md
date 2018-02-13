# w251_lyrics

## Setup

```
# as root or sudoer:
apt-get update
apt-get install tmux
apt-get install python
apt-get install python-pip
# (or install Anaconda)
pip install --upgrade pip
pip install scrapy ipython
# (or install scrapy with conda)
```

## Creating new spiders

```
git clone https://github.com/tpanza/w251_lyrics.git
cd w251_lyrics/
scrapy startproject mldb_scraper mldb_scraper
cd mldb_scraper
scrapy list
scrapy genspider mldb mldb.org
scrapy list
```

## Example run instructions

For the azlyrics.com scraper:

Run with logging status messages to local file (instead of stdout):

`scrapy crawl azlyrics --logfile azlyrics.log`

Run with saving cache to disk so that crawl can be interrupted and resumed:

`scrapy crawl azlyrics -s JOBDIR=crawls/azlyricsspider-1`

Combine the two options:

`scrapy crawl azlyrics --logfile azlyrics.log -s JOBDIR=crawls/azlyricsspider-1`

Recommend running the scraper(s) with `tmux` in case your SSH connection gets terminated. See <https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session> for details.

For more details, see:

 * https://doc.scrapy.org/en/latest/topics/logging.html
 * https://doc.scrapy.org/en/latest/topics/jobs.html

## S3 Bucket Access

```
pip install s3cmd
s3cmd --configure
# see Slack channel for credentials
s3cmd ls s3://w251lyrics-project
```
