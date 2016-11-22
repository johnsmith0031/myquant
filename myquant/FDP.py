import numpy as np

def kmeans(data, k=2, max_iter = 100, centers = None, tol = 1e-4):
    
    if centers is None:
        center_locs = []
        i = 0
        while i < k:
            loc = np.random.randint(0,len(data))
            while loc not in center_locs:
                center_locs.append(loc)
            i += 1
        centers = data[center_locs,:]*1
    
    for t in range(max_iter):
        
        dis = np.zeros((data.shape[0],centers.shape[0]))
        for i in range(centers.shape[0]):
                dis[:,i] = np.sum((data - centers[i,:])**2, axis=1)
        clusters = np.argmin(dis,axis=1)
        
        old_centers = centers*1
        for i in range(centers.shape[0]):
            centers[i,:] = np.mean(data[clusters == i,:], axis = 0)
                
        if np.sum((old_centers - centers)**2) < tol:
            break
    
    locs = np.repeat(False,len(data)*k).reshape((-1,k))
    for i in range(centers.shape[0]):
        locs[:,i] = clusters == i
    
    return locs
    
def FDP(data, k=2, max_iter = 100, threshold = 2, noise = 0.01):
    
    dis = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        dis[i,:] = np.sum((data - data[i,:])**2, axis=1)
    
    dc = np.percentile(dis,2)
    if dc == 0:
        dis = dis + np.random.rand(data.shape[0],data.shape[0])*noise
        dc = np.percentile(dis,2)
    p = np.sum(np.exp(-dis/dc),axis=1)
    
    q = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        loc = p > p[i]
        if np.sum(loc) == 0:
            q[i] = np.max(dis[i,:])
        else:
            q[i] = np.min(dis[i,loc])
    
    center_loc = np.argsort(p*q)[-k:]
    centers = data[center_loc,:]*1
    
    locs = kmeans(data,k = k,max_iter = max_iter,centers = centers)
    
    return locs
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    K = 10
    X = np.array([]).reshape((0,2))
    for i in range(K):
        center = np.random.randn(2)*5
        X = np.row_stack((X,np.random.randn(100,2)+center))
        
    #X = X[np.argsort(np.random.rand(len(X))),:]
    #plt.plot(X[:,0],X[:,1],'.')
    
    locs = FDP(X,K,100,2)
    for i in range(K):
        plt.plot(X[locs[:,i],0],X[locs[:,i],1],'.',alpha = 0.5)

    np.sum(locs,axis=0)









