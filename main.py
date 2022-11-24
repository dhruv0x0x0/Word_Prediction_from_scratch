import numpy as np
class read:
    def __init__(self, l):
       self.data=open("/Users/dhruva/PycharmProjects/rnnfromscratch/venv/rhyme.txt")
       self.text=self.data.read()
       vocab=list(set(self.text)) #Making vocab of unique characters from dataset
       self.map_chartoindex={y:x for (x,y) in enumerate(vocab)} #generating two way mapping for the vocab, will help in encoding
       self.map_indextochar={x:y for (x,y) in enumerate(vocab)}
       self.ts=len(self.text) #data size
       self.vs=len(vocab) #vocab size
       self.l=l #batch size
       self.c=0 #intialised pointer to char position in text to 0
    def next_batch(self):
       s=self.c # Batch start
       e=s+self.l #Batch end
       input=[self.map_chartoindex[char] for char in self.text[s:e]] #encoding batch
       target=[self.map_chartoindex[char] for char in self.text[s+1:e+1]] #encoding target batch
       self.c+=self.l
       if self.c+self.l+1>=self.ts:
           self.c=0
       return input,target
    def just_started(self):
        return self.c == 0
class rnn_from_scratch:
    def __init__(self, hs,vs,lr,l):
        self.hs=hs
        self.vs=vs
        self.lr=lr
        self.l=l
        self.Waa=np.random.uniform(-np.sqrt(1. /hs),np.sqrt(1. /hs),(hs,hs))
        self.Wax=np.random.uniform(-np.sqrt(1. /vs),np.sqrt(1. /vs),(hs,vs))
        self.Wya=np.random.uniform(-np.sqrt(1. / hs), np.sqrt(1. / hs), (vs, hs))
        self.ba=np.zeros((hs,1))
        self.by=np.zeros((vs,1))
        self.mWaa = np.zeros_like(self.Waa)
        self.mWax = np.zeros_like(self.Wax)
        self.mWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)


    def softmax(self, x): #return softmax output of an array
        p = np.exp(x - np.max(x))
        return p / np.sum(p)
    def forward(self,input,at_1):
        xt, at, y, os={},{},{},{}
        at[-1]=np.copy(at_1)
        for t in range(len(input)):
            xt[t]=np.zeros((self.vs,1))
            xt[t][input[t]]=1 #encoding of t'th char of input xt[t]=[0 0 0 0 0 1 0 0 0 0 0 0]
            at[t]= np.tanh(np.dot(self.Wax, xt[t]) + np.dot(self.Waa, at[t - 1]) + self.ba)
            os[t] = np.dot(self.Wya, at[t]) + self.by
            y[t]=self.softmax(os[t])
        return xt, at, y
    def backward(self, xt, at, pt, targets):
        dWaa=np.zeros_like(self.Waa)
        dWax=np.zeros_like(self.Wax)
        dWya=np.zeros_like(self.Wya)
        dba = np.zeros_like(self.ba)
        dby = np.zeros_like(self.by)
        #dat = np.zeros_like(at[0])
        da_next = np.zeros_like(at[0])
        for t in reversed(range(self.l)):
            dy = np.copy(pt[t])
            dy[targets[t]] -= 1  # backprop into y
            dWya += np.dot(dy, at[t].T)
            dby += dy
            da = np.dot(self.Wya.T, dy) + da_next  # backprop into h
            # backprop through tanh non-linearity
            da_rec = (1 - at[t] * at[t]) * da
            # print(np.shape(db))
            # print(np.shape(dhrec))
            # print(np.shape(hs[t].T))
            dba += da_rec
            # calculate dU and dW
            dWax += np.dot(da_rec, xt[t].T)
            dWaa += np.dot(da_rec, at[t - 1].T)
            # pass the gradient from next cell to the next iteration.
            da_next = np.dot(self.Waa.T, da_rec)
            # clip to mitigate exploding gradients
        for dparam in [dWax, dWaa, dWya, dba, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            return dWax, dWaa, dWya, dba, dby
    def loss(self, pt, targets):
        return sum(-np.log(pt[t][targets[t], 0]) for t in range(self.l))
    def update_model(self, dWax, dWaa, dWya, dba, dby):
        # parameter update with adagrad
        for param, dparam, mem in zip([self.Wax, self.Waa, self.Wya, self.ba, self.by],
                                      [dWax, dWaa, dWya, dba, dby],
                                      [self.mWax, self.mWaa, self.mWya, self.mba, self.mby]):
            mem += dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)  # measure for exploding gradient

    def sample(self, a, seed_ix, n):
        x = np.zeros((self.vs, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a) + self.ba)
            y = np.dot(self.Wya, a) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vs), p=p.ravel())
            x = np.zeros((self.vs, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
    def train(self, reader):
        iter_num = 0
        threshold = 0.01
        smooth_loss = -np.log(1.0 / reader.vs) * self.l
        while (smooth_loss > threshold):
            if reader.just_started():
                at_1 = np.zeros((self.hs, 1))
            inputs, targets = reader.next_batch()
            xt, at, pt = self.forward(inputs, at_1)
            dWax, dWaa, dWya, dba, dby = self.backward(xt, at, pt, targets)
            loss = self.loss(pt, targets)
            self.update_model(dWax, dWaa, dWya, dba, dby)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            at_1 = at[self.l- 1]
            if not iter_num % 500:
                sample_ix = self.sample(at_1, inputs[0], 200)
                print(''.join(reader.map_indextochar[ix] for ix in sample_ix))
                print("\n\niter :%d, loss:%f" % (iter_num, smooth_loss))
            iter_num += 1
    def predict(self, reader, start, n):

        # initialize input vector
        x = np.zeros((self.vs, 1))
        chars = [ch for ch in start]
        ixes = []
        for i in range(len(chars)):
            ix = reader.map_chartoindex[chars[i]]
            x[ix] = 1
            ixes.append(ix)

        a = np.zeros((self.hs, 1))
        # predict next n chars
        for t in range(n):
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a) + self.ba)
            y = np.dot(self.Wya, a) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vs), p=p.ravel())
            x = np.zeros((self.vs, 1))
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(reader.map_indextochar[i] for i in ixes)
        return txt

l = read(25)
rnn = rnn_from_scratch(hs=100, vs=l.vs, l=25, lr=1e-1)
rnn.train(l)
rnn.predict(l, 'king', 50)


