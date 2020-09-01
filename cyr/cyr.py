# -*- coding: UTF-8 -*-
import socket
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 27, kernel_size=6, stride=1, padding=2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(27, 54, kernel_size=4, stride=1, padding=2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(5*7*54, 800).double()
        self.fc2 = nn.Linear(800, 200).double()
        self.fc3 = nn.Linear(200, out).double()
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


drev = {'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'Dj': 'Ђ', 'DJ': 'Ђ', 'E': 'Е', 'I': 'И', 'J': 'Ј', 'K': 'К', 'L': 'Л', 'Lj': 'Љ', 'LJ': 'Љ', 'M': 'М', 'N': 'Н', 'Nj': 'Њ', 'NJ': 'Њ', 'O': 'О', 'P': 'П', 'R': 'Р', 'T': 'Т', 'U': 'У', 'F': 'Ф', 'H': 'Х', 'Dz': 'Џ', 'DZ': 'Џ', 'a': 'а', 'b': 'б', 'v': 'в', 'g': 'г', 'd': 'д', 'dj': 'ђ', 'e': 'е', 'i': 'и', 'j': 'ј', 'k': 'к', 'l': 'л', 'lj': 'љ', 'm': 'м', 'n': 'н', 'nj': 'њ', 'o': 'о', 'p': 'п', 'r': 'р', 't': 'т', 'u': 'у', 'f': 'ф', 'h': 'х', 'dz': 'џ'}
asci = [' ','a','b','v','g','d','dj','e','z','i','j','k','l','lj','m','n','nj','o','p','r','s','t','c','u','f','h','dz']
asci_len = len(asci)

predic = {
        ('s',0):'с',
        ('s',1):'ш',
        ('z',0):'з',
        ('z',1):'ж',
        ('c',0):'ц',
        ('c',1):'ћ',
        ('c',2):'ч',
        }

nets = {'s': Net(2).to(device), 'z': Net(2).to(device), 'c': Net(3).to(device)}

def load_nets(nets, nets_path):
    checkpoint = torch.load(nets_path, map_location=device)
    for k in nets.keys():
        nets[k].load_state_dict(checkpoint[k+'_net'])
        nets[k].eval()

critical_latin = ['s', 'z', 'c']
two_letter_letter = ['nj', 'lj', 'dz', 'dj']

def alpha_or_ws(x):
    if x == ' ':
        return True
    return x.isalpha()

def textToList(text):
    l = len(text)
    text += ' '
    caps = []
    huks = {}
    news = []
    orig = []
    i = 0
    while i < l:
        if not alpha_or_ws(text[i]):
            orig.append(text[i])
            i += 1
            continue
        if text[i] != ' ':
            caps.append(1 if text[i].isupper() else 0)
        if text[i:i+2].lower() in two_letter_letter:
            news.append(text[i:i+2].lower())
            orig.append(text[i:i+2].lower())
            i += 2
            continue
        if text[i].lower() in critical_latin:
            huks[i] = text[i].lower()
        news.append(text[i].lower())
        orig.append(text[i].lower())
        i += 1
    return caps, huks, news, orig

def seqToTensor(seq):
    tensor = [[0 for i in range(asci_len)] for x in range(len(seq))]
    for i, letter in enumerate(seq):
        tensor[i][asci.index(letter)] = 1
    return tensor

def huks_n_newsToData(huks, news):
    ins = {'s':[], 'z':[], 'c':[]}
    for i in huks.keys():
        t = huks[i]
        head = 10-i if 10-i > 0 else 0
        vector = [' ' for x in range(head)]
        vector += news[i-10 if i-10 > 0 else 0:i+11]
        vector += [' ' for x in range(21-len(vector))]
        vector = seqToTensor(vector)
        ins[t].append([vector])
    for i in ins.keys():
        ins[i] = torch.tensor(ins[i], device=device).double()
    return ins

def getOuts(nets, ins):
    outs = {}
    with torch.no_grad():
        for i in nets.keys():
            try: 
                outs[i] = nets[i](ins[i])
            except RuntimeError:
                pass
    return outs

def warn(xs):
    ys = sorted(xs, reverse=True)
    return 1 < (ys[0] - ys[1])

def percentage_prediction(outs):
    preds = {}
    percs = {}
    for i in outs.keys():
        preds[i] = np.apply_along_axis(np.argmax, 1, outs[i])
        percs[i] = np.apply_along_axis(warn, 1, outs[i])
    return preds, percs

def rev(caps, huks, preds, orig):
    l = len(orig)
    i = 0
    c = 0
    counters = {'s':0, 'z':0, 'c':0}
    while i < l:
        t = orig[i]
        if not t.isalpha():
            i += 1
            continue
        if t in critical_latin:
            orig[i] = predic[(t, preds[t][counters[t]])]
            counters[t] += 1
            if caps[c]:
                orig[i] = orig[i].upper()
            c += 1
            i += 1
        else:
            if caps[c]:
                orig[i] = t.upper()
            orig[i] = drev[orig[i]]
            c += 1
            i += 1
    return "".join(orig)
            

def convert(text):
    caps, huks, news, orig = textToList(text)
    ins = huks_n_newsToData(huks, news)
    outs = getOuts(nets, ins)
    preds, _ = percentage_prediction(outs)
    return rev(caps, huks, preds, orig)

def main():
    new_text = ""
    text = ""
    nets_path = os.path.expanduser("~") + "/.cyr_nets"
    file_name = ""
    if "-n" in sys.argv:
        if "-d" not in sys.argv:
            nets_path = sys.argv[sys.argv.index("-n")+1] 
    if "-s" in sys.argv:
        try: 
            load_nets(nets, nets_path)
        except Exception as e:
            print("Nets not found at "+nets_path + ".")
            sys.exit()
        
        s = socket.socket()
        host = socket.gethostname()
        port = 8082
        try:
            s.bind((host, port))
        except Exception as e:
            print("Daeomon already running or " + host + ":" + str(port) + " is already in use")
            sys.exit()
        s.listen(1)
        while True:
            c, _ = s.accept()
            data = str(c.recv(2048).decode())
            if data == "exit":
                c.close()
                s.close()
                sys.exit()
            else:
                c.send(convert(data).encode())
                c.close()
        sys.exit()
    if "-D" in sys.argv:
        os.system("nohup cyr -s -n " + nets_path + " &> /dev/null &")
        sys.exit()
    if "-K" in sys.argv:
        host = socket.gethostname()
        port = 8082
        client = socket.socket()
        try: 
            client.connect((host,port))
        except Exception as e:
            print('Make sure cyr daemon is running.')
            sys.exit()
        client.send("exit".encode())
        client.close()
        sys.exit()
    if "-h" in sys.argv:
        print(
"""cyr [-d] [-D] [-f FILE] [-h] [-i] [-K] [-n PATH] [-o FILE]
-d 
    use daemon instead of loading nets.
-D 
    start daemon. 
-f FILE
    specify file to read. reads from standard input by default.
-h 
    display this help and exit.
-i
    edit file in place.
-K 
    kill daemon.
-n PATH
    path to nets. default is $HOME/.cyr_nets.
-o FILE 
    specify output file. writes to standard output by default.""")
        sys.exit()
    if "-f" in sys.argv:
        file_name = sys.argv[sys.argv.index("-f")+1]
        try:
            with open(file_name, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print("File not found.")
            sys.exit()
        except IndexError:
            print("No file given, run with -h option for help.")
            sys.exit()
    else:
        for line in sys.stdin:
            text += line
    if "-d" in sys.argv:
        host = socket.gethostname()
        port = 8082
        client = socket.socket()
        try: 
            client.connect((host,port))
        except Exception as e:
            print('Make sure cyr daemon is running.')
            sys.exit()
        client.send(text.encode())
        new_text = client.recv(2048).decode()
        client.close()
    else:
        try:
            load_nets(nets, nets_path)
        except Exception as e:
            print('Error loading nets.')
            sys.exit()
        new_text = convert(text)
    if "-o" in sys.argv:
        try:
            with open(sys.argv[sys.argv.index("-o")+1], "w") as f:
                f.write(new_text)
        except Exception as e:
            print("No file given, run with -h option for help.")
            sys.exit()
    elif "-i" in sys.argv:
        if file_name == "":
            print("You must specify file with -f option.")
            sys.exit()
        else:
            try:
                with open(file_name, "w") as f:
                    f.write(new_text)
            except Exception as e:
                print("Error writting file.")
                sys.exit()

    else:
        print(new_text)



        

if __name__ == "__main__":
    main()
