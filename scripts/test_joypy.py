import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; np.random.seed(2)
import random; random.seed(2)
import joypy

# Sample data
df = pd.DataFrame({'var1': np.random.normal(70, 100, 500),
                   'var2': np.random.normal(250, 100, 500),
                   'group': random.choices(["bundle-%d"%(i+1) for i in range(20)], k = 500)})

fig, ax = joypy.joyplot(df, by = "group")

plt.show()