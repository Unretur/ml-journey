import numpy as np

import time 
list=list(range(1000000))
arr=np.arange(1000000)

start= time.time()
sum(list)
print("list time:",time.time()-start)

start=time.time()
np.sum(arr)
print("numpy time",time.time()-start)



