#!/bin/bash
set -ex

export USE_PODMAN=1
for test in square_iso square_linear cube_iso cube_linear cube_cylinder
do
    cd $test
    python run_test.py
    cd ../
done