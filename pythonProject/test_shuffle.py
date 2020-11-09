import random
random.seed(20)
data = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
x = [1,2,3,4,5,6]
p = random.shuffle(x)
def dataloader(feature,batch_size,shuffle=True):
    size = []
    list1 = [i for i in range(len(feature))]
    if(shuffle==True):
        random.shuffle(list1)

    i = 0
    while(i<len(feature)):
        temp = []
        k = 0
        while(k<batch_size):
            temp.append(list1[i+k])
            k=k+1
            if((i+k)>=len(feature)):
                break
        i = i+k
        size.append(temp)
    return size

print(dataloader(x,5))
# x = [i for i in range(6)]
# random.shuffle(x)
# print(x)