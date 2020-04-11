import requests
from bs4 import BeautifulSoup
import pandas as pd

list_art = list()
list_text = list()
list1 = [['id', 'text']]

headers = {'user-agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}

f = open('real_urls_vol2', "r")
f2 = open('true_texts_vol2', "a")
i = 1

for data in f:
     
    text_id = []
    j = 16
    while data[j].isdigit():
        j += 1
    
    text_id = data[:j]
    
    url = data[j:len(data) -1]
    list_text = []
    if "http" not in url:
        
        if "www." not in url:

            url = "https://www." + url
        else:

            url = "https://" + url
    
        
    print(url)
    if "longroom.com" in url:
        f2.write(str(text_id) + "-" + url +  "\n")
        
        f2.write("\n---------------------------------------------------------------------------------------------------------------\n")
        i += 1
        continue
    if "foodandwine.com" not in url and "southernliving.com" not in url and "realsimple.com/" not in url and "heightline.com/" not in url and "sportskeeda.com/" not in url and "travelandleisure.com/" not in url and "tvovermind.com/" not in url and "dreadcentral.com/" not in url and "rarolae.com/" not in url and  "metro.us/" not in url and "dailystar.co.uk/" not in url and "newbeauty.com" not in url and "manrepeller.com/" not in url and "pastemagazine.com/" not in url and "newsday.com/" not in url and "newsweek.com/" not in url and "overthemoon.com/" not in url and "turitmo.com" not in url and "worldstarsmag.com" not in url and "news.starbucks.com/" not in url and "entertainment.inquirer.net" not in url and "ebony.com" not in url and "digviral.com" not in url and "mindfood.com/" not in url and "newsday.com/" not in url and "huffpost.com" not in url and "newsweek.com" not in url and "tvovermind.com/" not in url and "hot1039.com" not in url and "cybersecurity-insiders.com" not in url and "purpleclover.com" not in url and "nbc.com" not in url and "newsweek.com" not in url and "health.com" not in url and "newbeauty.com" not in url and "rarolae.com" not in url and "cosmopolitan.com" not in url and "elledecor.com" not in url and "theblast.com" not in url and "vh1.com" not in url and "okayplayer.com" not in url and "mirror.co.uk" not in url and "time.com" not in url and "telegraph.co." not in url and "galoremag.com" not in url and "huffingtonpost.c" not in url and "parents.com" not in url and "people.com" not in url and "hellogiggles.com" not in url and "instyle.com" not in url and "/ew.com" not in url and "independent.co.uk" not in url and "omgcheckitout.com" not in url:
        cookies = {'enwiki_session': '17ab96bd8ffbe8ca58a78657a918558'}

        try :
            r = requests.get(url, cookies=cookies)
        except requests.exceptions.Timeout as tmout:
            continue
        except requests.exceptions.ConnectionError as conn:
            continue
        except KeyError:
            continue

        soup = BeautifulSoup(r.content, "html.parser")

        for p in soup.select('p'):
            list_text.append(p.get_text(strip=True))

        print("Text ready.")
        result = ' '.join(list_text)
        list1.append([text_id , result])
        f2.write(str(text_id) + "-" + url +  "\n")
        f2.write(result)
        f2.write("\n---------------------------------------------------------------------------------------------------------------\n")
        i += 1
        continue
        
    # if "longroom" in url:
    #     f2.write(str(text_id) + "-" + url +  "\n")
    #     f2.write("\n---------------------------------------------------------------------------------------------------------------\n")
    #     i += 1
    #     continue
    #url = "https://" + url
    try :
        s = requests.Session()
        # if "money.com" in url:
        #     print("ye")
        #     r = s.get("https://money.com",  headers=headers) 
        # else:
        #     r = s.get("https://time.com",  headers=headers) 
        r = s.get(url, headers=headers)
        cookies = r.cookies.get_dict()
        #print(s.cookies.get_dict())
        #print(type(r.cookies)
        r = s.get(url,  headers=headers , cookies=cookies)  
        
        #print(r.headers)
     
    except requests.exceptions.Timeout as tmout:    # Maybe set up for a retry, or continue in a retry loop
        # print(tmout)
        # archieve_url = "http://web.archive.org/cdx/search/cdx?url={}&output=json".format(url)

        # try:
        #     r = requests.get(url,   headers=headers, cookies=cookies)
        # except requests.exceptions.Timeout as t: 
        #     print(t)
        #    continue
        # except requests.exceptions.ConnectionError as c:
        #     print(c)
        continue

    except requests.exceptions.ConnectionError as conn:
        # print(conn)
        # archieve_url = "http://web.archive.org/cdx/search/cdx?url={}&output=json".format(url)

        # try:
        #     r = requests.get(url, headers=headers, cookies=cookies)
        # except requests.exceptions.Timeout as t: 
        #     print(t)
        #     continue
        # except requests.exceptions.ConnectionError as c:
        #     print(c)
        continue
    except KeyError:
        continue

    soup = BeautifulSoup(r.content, "html.parser")

    for p in soup.select('p'):
        list_text.append(p.get_text(strip=True))

    print("Text ready.")
    result = ' '.join(list_text)
    list1.append([text_id , result])
    f2.write(str(text_id) + "-" + url +  "\n")
    f2.write(result)
    f2.write("\n---------------------------------------------------------------------------------------------------------------\n")
    i += 1
        
f.close()
f2.close()

# df1 = pd.DataFrame(list1)
# df1.to_csv('real_texts.csv',sep=',',index = False ,header = False)
