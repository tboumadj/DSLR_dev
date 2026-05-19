import matplotlib.pyplot as plt
import numpy as np

def print_testmatplot():
# Fixing random state for reproducibility
    np.random.seed(19680801)

# make up some data in the open interval (0, 1)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))

# plot with various axes scales
    plt.figure(figsize=(9, 3))

#Plot cloud point
    data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100

    plt.subplot(121)
    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')

# logit
    plt.subplot(122)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)

# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

    plt.show()
