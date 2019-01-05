from bs4 import BeautifulSoup as soup
import requests
import re
import time

pattern = r'File:.* OriginalCircle.+'
url = 'http://leagueoflegends.wikia.com/wiki/Category:Champion_circles'
base_url = 'http://leagueoflegends.wikia.com'

og_links = []
while(True):
    html = requests.get(url).text
    sp = soup(html,'lxml')
    links = sp.find_all('a',{'class':'category-page__member-link'})
    og_links += [i['href'] for i in links if re.match(pattern, i.text)]
    next_pg = sp.find_all('a',{'class':'category-page__pagination-next wds-button wds-is-secondary'})
    time.sleep(0.5)
    if next_pg:
        url = next_pg[0]['href']
        print('next_page!')
    else:
        break

for i in og_links:
    html = requests.get(base_url + i).text
    sp = soup(html, 'lxml')
    img_link = sp.find('div',{'class':'fullImageLink'}).find('img')['data-src']
    champ_name = sp.find('h1',{'class':'page-header__title'}).text
    pattern = r'(.*) OriginalCircle.png'
    champ_name = re.findall(pattern, champ_name)[0]
    with open(f'./champion_icons/{champ_name}.jpg','wb') as f:
        content = requests.get(img_link).content
        f.write(content)
    print(img_link, champ_name)
