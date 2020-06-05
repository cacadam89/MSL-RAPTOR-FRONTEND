#!/usr/bin/env bash

cd utils_siammask/pyvotkit
python3 setup.py build_ext --inplace
cd ../../

cd utils_siammask/pysot/utils/
python3 setup.py build_ext --inplace
cd ../../../
