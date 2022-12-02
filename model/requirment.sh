#!/bin/bash
echo "strat pip back"

pip install mmcv_full-1.2.7+torch1.7.0+cu110-cp38-cp38-manylinux1_x86_64.whl
#cd mmcv-1.2.7
#pip install -r requirements.txt
#python setup.py build_ext
#python setup.py develop
#echo "finished pip mmcv"
#cd ..
#pip install -r requirements.txt
#pip install -e . --user

pip install mmsegmentation==0.11.0
echo "finished pip -r"