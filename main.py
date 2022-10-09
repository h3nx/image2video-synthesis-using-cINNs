
import numpy as np


if __name__ == '__main__':
    count = 5
    seq = 17
    #asd = list(range(count))
    #asd.extend([0]*(seq-count))




    asd = np.arange(0, count)
    asd = np.append(asd, [count-1]*(seq-count))
    print(len(asd),asd)


    tru = 9
    r = []
    for i in range(tru):
        r.append(i)
    for i in range(seq - tru):
        r.append(0)

    print("tru")
    print(r)