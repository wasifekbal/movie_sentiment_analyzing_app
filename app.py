import re
import numpy as np
import pandas as pd
from flask import Flask,request,render_template,redirect,session
import requests
from bs4 import BeautifulSoup as bf
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.corpus import stopwords


cv=CountVectorizer(max_features=1000)
ps=PorterStemmer()

model = joblib.load('model.pkl')
db = joblib.load('movie_tuple')

def review_scrapper(imdb_id):
    reviews_df = pd.DataFrame(columns=['rating','title','username','review_date','review_text'])
    title_url = 'https://www.imdb.com/title/'
    res = requests.get(url=title_url+imdb_id+'/reviews').text
    soup = bf(res,'html5lib')
    review_boxs = soup.find_all(name='div', attrs={'class':'review-container'})
    for i in review_boxs:
        try:
            rate = int(i.find_all(name='span')[1].text)
        except:
            rate = 'nan'
        title = i.find(name='a',class_='title').text.strip()
        username = i.find(name='div', class_='display-name-date').find('a').text
        review_date = i.find(name='div', class_='display-name-date').find(class_='review-date').text
        review_text = i.find(name='div', class_='text show-more__control').text
        temp = {'rating':rate,'title':title,'username':username,'review_date':review_date, 'review_text':review_text}
        reviews_df = reviews_df.append(temp,ignore_index=True)
    return reviews_df
def info_scraper(imdb_id):
    title_url = 'https://www.imdb.com/title/'
    res = requests.get(url=title_url+imdb_id).text
    soup = bf(res,'html5lib')
    rating = float(soup.find(name='span', attrs={"itemprop":"ratingValue"}).text)
    x = soup.find(name='h1', attrs={'class':''}).text
    title = x.split("\xa0")[0] +' '+ x.split("\xa0")[1].strip()
    info = soup.find(name='div',class_='subtext').text
    duration = info.split("|\n")[0].strip()
    genres = info.split("|\n")[1].split("\n")[0]+info.split("|\n")[1].split("\n")[1]
    yor = info.split("\n")[-2]
    img_url = soup.find(name='div', class_='poster').find(name='img').attrs['src']
    hd = 'UX650_CR1,0,680,1000_AL__QL50.jpg'
    img_url = img_url[:-27] + hd ## for heroku '27', for local '32'
    try:
        text = soup.find(name='div', attrs={'class': 'inline canwrap'}).find('p').text
        storyline = ''
        for i in text.strip().split("\n"):
            storyline = storyline + " " + i
    except:
        storyline = 'Not found any storyline for this move.'

    return [rating,title,duration,genres,yor,img_url,storyline]

def spacial_char_cleaner(string):
    ptrn = re.compile("\'.")
    ptrn2 = re.compile(r"\n")
    text = re.sub(pattern=ptrn,repl='',string=string)
    text = re.sub(pattern=ptrn2,repl=' ',string=text)
    return text
def punctuation_remover(text):
    x=''
    for i in text:
        if (i.isalnum()) or i==' ':
            x=x+i
        else:
            x=x + ''
    return x
def remove_stopwords(string):
    text = []
    for i in string.split():
        if i not in stopwords.words('english'):
            text.append(i)
    x = ' '.join(text)
    text.clear()
    return x
def stem_words(string):
    text=[]
    for i in string.split():
        text.append(ps.stem(i))
    return ' '.join(text)
def full_cleaning(string):
    temp = spacial_char_cleaner(string)
    temp = punctuation_remover(temp)
    temp = temp.lower()
    temp = remove_stopwords(temp)
    temp = stem_words(temp)
    return temp

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search',methods=['POST'])
def search():
    q = (request.form.get('movie')).lower()
    if q != '':
        p = re.compile(f"{q}?")
        imdb_ids = []
        movie_names = []
        yrs = []
        for i in db:
            check = i[1].lower()
            if re.search(p, check):
                imdb_ids.append(i[0])
                movie_names.append(i[1])
                yrs.append(i[2])
        l=list(range(len(imdb_ids)))
        return render_template('result.html',imdb_ids=imdb_ids,movie_names=movie_names,yrs=yrs,length=l);
    else:
        return redirect('/')

@app.route('/predict',methods=['POST'])
def predict():
    movie_id = request.form.get('movie_id')
    movie_info = info_scraper(str(movie_id))
    review_df = review_scrapper(str(movie_id))
    l = list(range(len(review_df)))
    review_df['cleaned_reviewed_test'] = review_df['review_text'].apply(full_cleaning)
    try:
        X = cv.fit_transform(review_df['cleaned_reviewed_test']).toarray()
        y_pred = model.predict(X)
        review_df.insert(column='prediction', value=y_pred, loc=review_df.shape[1])
        perc = int((review_df['prediction'].value_counts()[1] / 25) * 100) / 10
        flag = 0
        return render_template('predict.html', movie_info=movie_info, review_df=review_df, length=l, perc=perc,flag=flag)
    except:
        flag = 1
        return render_template('predict.html',movie_info=movie_info,flag=flag)

if __name__ == '__main__':
    app.run(debug=True)