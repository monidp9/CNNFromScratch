from utility import get_nn_type
import cv_main
import fc_main

nn_type = get_nn_type()

if nn_type == 'fc_nn':
	fc_main.run()
else:
	cv_main.run()
