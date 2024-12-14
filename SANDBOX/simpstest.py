import numpy as np
from scipy.integrate import simpson

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Perform integration
result = simpson(y, x= x)
print(result)
