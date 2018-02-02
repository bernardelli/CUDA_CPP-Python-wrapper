# -*- coding: utf-8 -*-
import sys
sys.path.append('./build/Debug')

import numpy as np
import vectorlib as vl

a = vl.Vector(np.array([1, 2, 3]));

b = vl.Vector(np.array([4, 5, 6]));

print(a)
print(b)
c = a + b;
print(c)
