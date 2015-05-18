#!/usr/bin/env python

import os


TESTS=['test_plotfenics', 'test_prior']


for test in TESTS:
    os.system('python -m unittest -v Test.{0}'.format(test))

