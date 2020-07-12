# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/3 9:10  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from scipy import stats


if __name__ == '__main__':
  a = np.array([[2, 2], [2, 2], [3, 3]])
  b = np.array([2, 2])
  print(b.reshape([-1, 1]))