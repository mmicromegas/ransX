# https://docs.sympy.org/dev/modules/physics/vector/api/fieldfunctions.html#curl
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl

# https://stackoverflow.com/questions/30378676/calculate-curl-of-a-vector-field-in-python-and-plot-it-with-matplotlib
import scipy as sp
import numdifftools as nd

# R = ReferenceFrame('R')

# v1 = R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
# print(curl(v1,R))

# v2 = R[0]*R[1]*R[2]*R.x
# print(curl(v2,R))

# v3 = 2.*R.x + 5.*R.y + 8*R.z
# print(curl(v3,R))


def h(x):
    return sp.array([3*x[0]**2,4*x[1]*x[2]**3, 2*x[0]])

def curl(f,x):
    jac = nd.Jacobian(f)(x)
    return sp.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])

x = sp.array([1,2,3])
curl(h,x)


