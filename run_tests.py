#!/usr/bin/env python

import os


TESTS=['test_plotfenics', 'test_datafenics', 'test_operatorfenics_Mass',
'test_operatorfenics_Helmholtz']


for test in TESTS:
    os.system('python -m unittest -v Test.{0}'.format(test))

