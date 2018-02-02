# -*- coding: utf-8 -*-
import sys
sys.path.append('./build/Debug')

import vectorlib as vl

a = vl.Vector([1, 2, 3]);
b = vl.Vector([4, 5, 6]);

print(a)
print(b)
c = a + b;
print(c)
