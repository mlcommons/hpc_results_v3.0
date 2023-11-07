#!/bin/bash

mpirun -n 4 --tag-output python examples/cosmo.py  |& tee $1
