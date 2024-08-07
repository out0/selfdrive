#! /bin/sh
python3 -m unittest discover -v model/
python3 -m unittest discover -v data/
python3 -m unittest discover -v motion/
python3 -m unittest discover -v utils/
#python3 -m unittest discover -v slam/
