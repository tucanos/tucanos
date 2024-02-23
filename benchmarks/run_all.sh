#!/bin/bash
set -ex

for test in square_iso square_linear cube_iso cube_linear cube_cylinder
do
    cd $test
    python run_test.py
    cd ../
done