# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup as bs
import urllib
import torch
import torch.autograd as ag

first = "http://www.politika.rs/scc/clanak/667"
last =  "http://www.politika.rs/scc/clanak/450255"
clanak = "http://www.politika.rs/scc/clanak/"

dtype = torch.float
device = torch.device("cpu")

dick = {' ':' ','А':'A','Б':'B','В':'V','Г':'G','Д':'D','Ђ':'Dj','Е':'E','Ж':'Z','З':'Z','И':'I','Ј':'J','К':'K','Л':'L','Љ':'Lj','М':'M','Н':'N','Њ':'Nj','О':'O','П':'P','Р':'R','С':'S','Т':'T','Ћ':'C','У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'C','Џ':'Dz','Ш':'S','а':'a','б':'b','в':'v','г':'g','д':'d','ђ':'dj','е':'e','ж':'z','з':'z','и':'i','ј':'j','к':'k','л':'l','љ':'lj','м':'m','н':'n','њ':'nj','о':'o','п':'p','р':'r','с':'s','т':'t','ћ':'c','у':'u','ф':'f','х':'h','ц':'c','ч':'c','џ':'dz','ш':'s'}
asci = [' ','a','b','v','g','d','dj','e','z','i','j','k','l','lj','m','n','nj','o','p','r','s','t','c','u','f','h','dz']
asci_len = len(asci)
models = {'с': 0, 'ш': 0, 'з': 1, 'ж': 1, 'ц': 2, 'ћ':2, 'ч': 2}
critical = ['с', 'ш', 'з', 'ж', 'ц', 'ћ', 'ч']

outTensor = {
        'с': [1, 0],
        'ш': [0, 1],
        'з': [1, 0],
        'ж': [0, 1],
        'ц': [1, 0, 0],
        'ћ': [0, 1, 0],
        'ч': [0, 0, 1],
        }

outTensor2 = {
        'с': 0,
        'ш': 1,
        'з': 0,
        'ж': 1,
        'ц': 0,
        'ћ': 1,
        'ч': 2,
        }

def seqToTensor(seq):
    tensor = [[0 for i in range(asci_len)] for x in range(len(seq))]
    for i, letter in enumerate(seq):
        tensor[i][asci.index(letter)] = 1;
    return tensor


def getSoup(url):
    f = urllib.request.urlopen(url)
    html = f.read()
    f.close()
    soup = bs(html, features="html5lib")
    return soup

def getAlphaText(soup):
    divs = soup.find_all('div')
    text = None
    for d in divs:
        if 'class' in d.attrs:
            if d['class'] == ['article-content', 'mt3', 'mb3']:
                text = d.text
    if text == None:
        return None
    f = dick.keys()
    return " ".join("".join([' ' if w not in f else w for w in text]).split()).lower()

def UrlToText(url):
    return getAlphaText(getSoup(url))

def numToClanak(n):
    html = clanak + str(n)
    return UrlToText(html)

def textToFeed(text):
    kuke = {}
    new_text = []
    l = len(text)
    for i in range(l):
        if text[i] in critical:
            kuke[i] = text[i]
        new_text.append(dick[text[i]])
    Ds_in = []
    Ds_out = []
    Dz_in = []
    Dz_out = []
    Dc_in = []
    Dc_out = []
    for i in kuke.keys():
        t = kuke[i]
        outT = outTensor2[t]
        head = 10-i if 10-i > 0 else 0
        vector = [' ' for x in range(head)]
        vector += new_text[i-10 if i-10 > 0 else 0:i+11]
        vector += [' ' for x in range(21-len(vector))]
        vector = seqToTensor(vector)
        if models[t] == 0:
            Ds_in.append([vector])
            Ds_out.append(outT)
        elif models[t] == 1:
            Dz_in.append([vector])
            Dz_out.append(outT)
        else:
            Dc_in.append([vector])
            Dc_out.append(outT)
    #return torch.tensor(Ds_in, device=device, dtype=dtype).double(), torch.tensor(Ds_out, device=device, dtype=dtype).double(), torch.tensor(Dz_in, device=device, dtype=dtype).double(), torch.tensor(Dz_out, device=device, dtype=dtype).double(), torch.tensor(Dc_in, device=device, dtype=dtype).double(), torch.tensor(Dc_out, device=device, dtype=dtype).double()
    return ag.Variable(torch.tensor(Ds_in, device=device).double()), ag.Variable(torch.tensor(Ds_out, device=device)), ag.Variable(torch.tensor(Dz_in, device=device).double()), ag.Variable(torch.tensor(Dz_out, device=device)), ag.Variable(torch.tensor(Dc_in, device=device).double()), ag.Variable(torch.tensor(Dc_out, device=device))

def numToFeed(num):
    return textToFeed(numToClanak(num))

Ds_in, Ds_out, Dz_in, Dz_out, Dc_in, Dc_out = numToFeed(450443)
