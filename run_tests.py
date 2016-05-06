#!/usr/bin/env python

import os


TESTS=['unittest_plotfenics', 'unittest_prior', 'unittest_linalg', \
'unittest_operatorfenics_Helmholtz', 'unittest_operatorfenics_Mass', \
'unittest_datafenics']


for test in TESTS:
    os.system('python -m unittest -v Test.{0}'.format(test))
    print "\n\n"

