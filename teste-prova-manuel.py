import numpy as np
from scipy.spatial import distance
np.random.seed(26112000)

#Questão 01
def Q01(C, p, param):

    Xc = []

    for i in range(C):
        Xc.append(np.random.normal(param[0][i], param[1][i], (np.random.randint(50),p)))
    
    Xnp = np.vstack((Xc))

    return Xnp

mu = [
    [12,5],
    [8.5,7],
    [1,4]
]
covar = [
    [0.954,0.2],
    [0.37,0.87],
    [0.123,0.234]
]

Q1 = Q01(3,2, param = (mu, covar))

print(Q1)


# Questão 02
def Q02(Xnp, k):

    Wkp = []

    for j in range(k):
        
        Wkp.append(Xnp[np.random.randint(len(Xnp[:, 1]), size = 1)])

    Wkp = np.vstack(Wkp)
    
    return Wkp
Q2 = Q02(Q1, 4)
print(Q2)
print()
# Questão 03
def Q03(Xnp, Wkp):
    
    Dnk = np.zeros((len(Xnp), len(Wkp)))

    for i in range(len(Xnp)):
        for j in range(len(Wkp)):
            Dnk[i,j] = np.sqrt(np.sum((Xnp[i,:] - Wkp[j,:])**(2)))

    return Dnk

Q3 = Q03(Q1, Q2)

print(Q3)
print()

# Questão 04
Q4 = np.argmin(Q3, axis = 1)
print(Q4)
print()

# Questão 05
print()
def Q05(Xnp, Wkp, Dnk, c):
    k = np.argmin(Dnk, axis = 1)
    for i in range(c):
        X = Xnp[k == i,:]
        Wkp[i,:] = np.apply_along_axis(func1d = np.mean, arr = X, axis = 0)
    
    return Wkp

Q5 = Q05(Q1, Q2, Q3, 3)

print(Q5)

# Questão 06
print()
class KMeans:
    def __init__(self, k, tmax):
        self.k, self.tmax = k, tmax
    def fit(self,X, k):
        W = Q02(X,k)
        for i in range(self.tmax):
            dists = Q03(X, W)
            new_W = Q05(X, W, dists, k)
        
        return new_W
X = Q1
k = 5
tmax = 1000
km = KMeans(k, tmax)

print(km.fit(X, k))

# Questão 07
print()
class KMeans:
    def __init__(self, k, tmax):
        self.k, self.tmax = k, tmax

    def fit(self,X, k):
        W = Q02(X,k)
        for i in range(self.tmax):
            dists = Q03(X, W)
            new_W = Q05(X, W, dists, k)
        
        self.W = W

        return new_W

    def predict(self,X):
        dists = Q03(X, self.W)
        return np.argmin(dists , axis = 1)
    
X = Q1
k = 5
tmax = 100
km = KMeans(k, tmax)
X1 = Q01(2,2, param = ((0,1), (0,1)))

print(km.fit(X,k) , km.predict(X1))

# Questão 08
print()
class KMeans:
    def __init__(self, k, tmax):
        self.k, self.tmax = k, tmax

    def fit(self,X, k):
        W = Q02(X,k)
        for i in range(self.tmax):
            dists = Q03(X, W)
            new_W = Q05(X, W, dists, k)
        
        self.W = W

        return new_W

    def predict(self,X):
        dists = Q03(X, self.W)
        return np.argmin(dists , axis = 1)

    def score(self, X):
        d = Q03(X, self.W)
        d = np.apply_along_axis(min, 1, d)
        return np.sum(d)
X = Q1
k = 5
tmax = 100
km = KMeans(k, tmax)
X1 = Q01(2,2, param = ((0,1), (0,1)))
X2 = Q01(2,2, param = ((0,1), (0,1)))
km.fit(X,k)
print(km.fit(X,k), km.predict(X1), km.score(X2))

print()