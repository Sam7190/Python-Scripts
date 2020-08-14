import numpy as np
default = ['Hit Points', 'Agility', 'Stability', 'Cunning', 'Attack', 'Technique', 'Def - Physical', 'Def - Wizard', 'Def - Elemental', 'Def - Trooper']

def randofsum_unbalanced(s, n):
    r = np.random.rand(n)
    a = np.array(np.round((r/np.sum(r))*s,0),dtype=int)
    while np.sum(a) > s:
        a[np.random.choice(n)] -= 1
    while np.sum(a) < s:
        a[np.random.choice(n)] += 1
    return a

def randofsum(s, n):
    return np.random.multinomial(s,np.ones(n)/n,size=1)[0]
    
def genstats(s, atr=None):
    if type(s) is list:
        s = np.random.choice(np.arange(s[0], s[1]+1))
    if atr is None:
        lbls = default
    elif type(atr) is int:
        lbls = default[:atr] + default[(atr+1):]
    else:
        lbls = atr
    conditions = [None, None, None]
    n = len(lbls)
    a = randofsum(s, n)
    if ('Hit Points' in lbls):
        hi = lbls.index('Hit Points')
        conditions[0] = 'a[hi]==0'
    if ('Attack' in lbls) and ('Stability' in lbls):
        ai, si = lbls.index('Attack'), lbls.index('Stability')
        conditions[1] = 'a[si]>a[ai]'
    if ('Attack' in lbls):
        ai = lbls.index('Attack')
        conditions[2] = 'a[ai]==0'
    while True:
        a = randofsum(s, n)
        breakout = True
        for i in range(len(conditions)):
            if (conditions[i] is not None) and eval(conditions[i]):
                breakout = False
                break
        if breakout: break
    print(f"---- Combat Level: {sum(a)} ----")
    for i in range(len(a)): print(f'{lbls[i]}: {a[i]}')
    
genstats(50)