import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
import csv

tweets = []

with open("twitterbot\TweetsElonMusk.csv", 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        tweets.append(row['tweet'])

text = '\n'.join(tweets)

print("text extracted")
chars=sorted(list(set(text)))
vocab_size=len(chars)

stoi={s:i for i, s in enumerate(chars)}
itos={i:s for i,s in enumerate(chars)}
encode=lambda s: [stoi[j] for j in s]
decode=lambda i: ''.join([itos[s] for s in i])

data=torch.tensor(encode(text),dtype=torch.long)
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]


#------ Hyperparameters--------

n_embd=384
blocksize=256
batch_size=64
learning_rate=3e-4
n_head=8
eval_iters = 200
max_iters = 5000
eval_interval = 500
n_layer=6


torch.manual_seed(1337)




def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-blocksize,(batch_size,))

    x=torch.stack([data[i:i+blocksize] for i in ix])
    y=torch.stack([data[i+1:i+blocksize+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(blocksize,blocksize)))
        

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)

        wei=q@k.transpose(-2,-1) * k.shape[-1]**-0.5
        
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        attention=wei@v
        
        return attention
    


class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,head_size):
        super().__init__()
        self.lm_heads=nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(head_size * num_head, n_embd)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.lm_heads],dim=-1)
        out=self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.f_seq=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
        )

    def forward(self,x):
        return self.f_seq(x)
    

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.mht=MultiHeadAttention(n_head,head_size)
        self.ff=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x=x+self.mht(self.ln1(x))
        x=x+self.ff(self.ln2(x))
        return x
        

class ElonGPT(nn.Module):
    def __init__(self,vocab_size,n_embd):
        super().__init__()
        self.char_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(blocksize,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.finalLayerNorm=nn.LayerNorm(n_embd)
        self.l_f=nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_embd=self.char_embedding_table(idx)
        pos_embd=self.position_embedding_table(torch.arange(T,device=device))
        x=tok_embd+pos_embd
        x=self.blocks(x)
        x=self.finalLayerNorm(x)
        logits=self.l_f(x)
        if targets is None:
            loss=None
        
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)

            loss=F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -blocksize:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

        # return x

model=ElonGPT(vocab_size,n_embd)
m=model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)


for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    X,Y=get_batch('train')
    logits,loss=model(X,Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print(loss)

context = torch.zeros((1, 1), dtype=torch.long,device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))