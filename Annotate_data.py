import numpy as np

def next_batch(index, batch_size):
    result = np.reshape(np.arange(batch_size*184*216*18), (batch_size, 184, 216, 18)) #184x216x18
    print('annotate shape:', result.shape)
    # print('annotate data:\n', result)
    return result

tr=[[False,False ,False ,False  ,True ,False],[ True, False, False , False, False ,False]
  ,[True ,False,  True , True , True  ,True]]
print(np.sum(tr))