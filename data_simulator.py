import random
import matplotlib.pyplot as plt
import numpy as np
import math

def getRandomPointInCircle(num, radius, centerx, centery, noise):
    samplePoint = []
    for i in range(num):
        theta = random.random() * 2 * np.pi
        r = radius ** 2
        x = math.cos(theta) * (r ** 0.5) + centerx + np.random.randn() * noise[0, 0] ** 0.5
        y = math.sin(theta) * (r ** 0.5) + centery + + np.random.randn() * noise[1, 1] ** 0.5
        samplePoint.append((x, y))
        plt.plot(x, y, '*', color="blue")

    return samplePoint

if __name__ == "__main__":
    num = 20
    radius = 10
    centerx,centery = 10, 10
    m_sim = np.diag([0.2, 0.2]) ** 2
    samp = getRandomPointInCircle(num, radius, centerx, centery, m_sim)
    print("sample point" , samp)
    plt.legend()
    plt.show()