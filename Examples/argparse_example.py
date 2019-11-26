# python argparse.py /home/villa/ -snapnums 0 1 2 3 --long_ids
# for help type: python argparse.py -h
import argparse
import numpy as np
import sys,os

parser = argparse.ArgumentParser(description="description of routine")

# non-optional arguments
parser.add_argument("snapdir", help="folder where the groups_XXX folder is")

# optional arguments
parser.add_argument("-cx1", type=int, default=0, 
                    help="column x in file 1, default 0")
parser.add_argument("-snapnums", nargs='+', type=int, help="groups number")

parser.add_argument("--swap", dest="swap", action="store_true", default=False, 
                    help="False by default. Set --swap for True")

parser.add_argument("--SFR", dest="SFR", action="store_true", default=False, 
                    help="False by default. Set --SFR for True")

parser.add_argument("--long_ids", dest="long_ids", action="store_true", 
                    default=False, help="False by default. Set --long_ids for True")

args = parser.parse_args()

print(args.snapdir)
print(args.snapnums)
print(args.swap)
print(args.SFR)
print(args.long_ids)