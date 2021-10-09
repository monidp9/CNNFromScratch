from utility import get_nn_type
import cv_main
import fc_main
import os

nn_type = get_nn_type()

os.system('clear')

if nn_type == 'fc_nn':
	fc_main.run()
else:
	cv_main.run()
