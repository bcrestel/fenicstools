#!/usr/bin/env python

import os


TESTS=['test_plotfenics', 'test_regularization']


for test in TESTS:
    os.system('python -m unittest -v Test.{0}'.format(test))

