'''
只适用于二分类，多分类写不出来
'''
import numpy as np
def f1(y_pred,y_true,classifications):
    '''
    标签值为0和1
    :param y_hat: 预测值
    :param y_true: 真实值
    :param threshold: 阈值
    :return:
    '''
    # TP：真阳性，FP: 假阳性 FN：假阴性，TN 真阴性
    epsilon = 1e-7
    P = []
    R = []
    score = []
    for i in range(classifications):
        TP,FP,FN=0,0,0
        for k in range(len(y_pred)):
            if((y_pred[k]==i) and (y_true[k]==i)):
                TP = TP+1
            elif((y_true[k]!=i) and (y_pred[k]==i)):
                FP = FP+1
            elif((y_true[k]==i) and (y_pred[k]!=i)):
                FN = FN+1
            else:
                pass
        temp1 = TP/(TP+FP+epsilon)#P值
        temp2 = TP/(TP+FN+epsilon)#R值
        P.append(temp1)
        R.append(temp2)
        temp3 = 2*temp1*temp2/(temp1+temp2+epsilon)
        score.append(temp3)
    return np.mean(score)




# from sklearn.metrics import f1_score
# from collections import Counter
# y_pred = [0, 1, 1, 1, 2, 2]
# y_true = [0, 1, 0, 2, 1, 1]
# print(f1(y_pred,y_true,3))
# # print(f1(y_pred,y_true))
# print(f1_score(y_pred, y_true, average='macro'))