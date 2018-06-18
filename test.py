
import numpy as np

if __name__ == '__main__':
    nums = np.array([1, 2, 3, 4, 5])
    d = np.arange(len(nums))
    np.random.shuffle(d)
    print(nums[d])
