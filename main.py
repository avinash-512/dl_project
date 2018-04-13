import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import json
import random
import argparse


def prepare_data(filepath="harry1.txt",load=None,seq_length=100):
    if load==None:
        raw_text = open(filepath).read()
        raw_text = raw_text.decode("utf-8")
        raw_text = raw_text.encode("ascii","ignore")
        raw_text = raw_text.lower()
        chars = sorted(list(set(raw_text)))
        with open("chars.json","w") as f:
            f.write(json.dumps(chars,indent=2))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        n_chars = len(raw_text)
        n_vocab = len(chars)
        print "Total Characters: ", n_chars
        print "Total Vocab: ", n_vocab
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        print "Total Patterns: ", n_patterns
        # reshape X to be [samples, time steps, features]
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        np.save("X_train.npy",X)
        np.save("Y_train.npy",y)
    else:
        X = np.load("X_train.npy")
        y = np.load("Y_train.npy")
    return X,y

class story:
    def __init__(self,n_layer=256,dropout=0.2,seq_length=100):
        self.chars = json.loads(open("chars.json","r").read())
        self.char_to_int = dict((c,i) for i,c in enumerate(self.chars))
        self.int_to_char = dict((i,c) for i,c in enumerate(self.chars))
        self.n_vocab = len(self.chars)       
        self.model = Sequential()
        self.model.add(LSTM(n_layer, input_shape=(seq_length, 1),return_sequences=True))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(self.n_vocab, activation='softmax'))
        

    def train(self,X,y,epochs=10,batch_size=128):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        filepath="weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    def predict(self,text,model_weights="",n_chars=1):
        if model_weights == "":
            print "Provide path for the model weights"
            return
        self.model.load_weights(model_weights)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        pattern = self.process(text)
        y = ""
        result = ""
        while(len(result)<n_chars):
            x = np.reshape(pattern,(1,len(pattern),1))
            x = x/float(self.n_vocab)
            pred = self.model.predict(x,verbose=0)
            print "Hello"
            index = np.argmax(pred)
            y = self.int_to_char[index]
            pattern.append(index)
            pattern = pattern[1:]
            result += y
        return result
    
    def process(self,text):
        input = []
        text = text.replace("\t"," ")
        for c in text:
            input.append(self.char_to_int[c])
        return input

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--generate', action="store", dest="generate")
    parser.add_argument('--epochs', action="store", dest="epochs", type=int)
    parser.add_argument('--n_layer', action="store", dest="n_layer", type=int)
    parser.add_argument('--batch_size', action="store", dest="batch_size", type=int)
    parser.add_argument('--dropout', action="store", dest="dropout", type=float)
    parser.add_argument('--seq_length', action="store", dest="seq_length", type=float)
    
    results = parser.parse_args()   
    if train:
        epochs = 30
        n_layer = 256
        batch_size = 128
        dropout = 0.2
        seq_length = 100

        if results.epochs!= None:
            epochs = results.epochs
        if results.n_layer!= None:
            n_layer = results.n_layer
        if results.batch_size!= None:
            batch_size = results.batch_size
        if results.dropout!= None:
            dropout = results.dropout
        if results.seq_length!= None:
            seq_length = results.seq_length
        X_train,y_train = prepare_data("alice.txt")
        m = story(n_layer,dropout,seq_length)
        m.train(X_train,y_train,epochs,batch_size)
    else:
        raw_text = open("alice.txt","r").read()
        raw_text = raw_text.decode("utf-8")
        raw_text = raw_text.encode("ascii","ignore")
        raw_text = raw_text.lower()
        rand_int = random.randint(0,len(raw_text)-100)
        x = raw_text[rand_int:rand_int+100]
        m = story()
        ans = m.predict(x,"weights",200)
        print "Seed:" ,x
        print "Predicted:", ans