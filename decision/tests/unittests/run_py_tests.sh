#! /bin/bash

python3 -m unittest discover -v .
python3 -m unittest discover -v model/ &&
python3 -m unittest discover -v motion/ &&
python3 -m unittest discover -v planner/
#python3 -m unittest discover -v slam/