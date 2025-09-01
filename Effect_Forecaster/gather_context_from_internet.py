from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


def GatherContext(query,num_articles=5):
    chrome_options=Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    
    driver=webdriver.Chrome(options=chrome_options)

    articles=[]
    transformed_query=query.replace(" ","+")
    transformed_query+="news"

    url=f"https://www.google.com/search?sca_esv=0341e61adf0435df&sxsrf=AE3TifOV_ZKMb_TNjgXvvF1MYg47uecPIw:1756732435305&q={transformed_query}&tbm=nws&source=lnms&fbs=AIIjpHxU7SXXniUZfeShr2fp4giZrjP_Cx0LI1Ytb_FGcOviEreERTNAkkP8Y6EXltYTGWs9RGaEMfZ2dZFFrbmM-rnqDh6hIMjQfCAhrLJz0ZmMZpUjCqTWcnwxm3CIjs0Garqy8TVVj4tPonIrL4EWNMu8rOvh-xrP14qzx5YmNgCEsUOICOOCcUp1ctDbQqNbM_igTLTjqP-D7xm4lWV9xDiIztrnlg&sa=X&ved=2ahUKEwizjZam0rePAxXwVmwGHZarId4Q0pQJegQIFRAB&biw=1854&bih=927"
    
    driver.get(url)
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    driver.implicitly_wait(10)

    response=driver.page_source
    soup=BeautifulSoup(response,'html.parser')

    element=soup.find_all("div",class_="n0jPhd ynAwRc MBeuO nDgy9d")
    for ind,elem in enumerate(element):
        if ind>=num_articles:
            break

        articles.append(elem.get_text())

    driver.quit()
    return articles

if __name__=="__main__":
    query="Trump and Modi"
    articles=GatherContext(query)
    print(articles)