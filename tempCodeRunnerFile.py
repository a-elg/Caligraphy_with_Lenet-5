import time 

# \033[K
limite=1000
for i in range(1,limite+1):

    print("|<","-"*int(i*100/limite)," "*(100-int(i*100/limite)),">|",end="\r")
    time.sleep(0.1)
print("")