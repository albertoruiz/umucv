from umucv.stream import sourceArgs
import sys
import argparse

parser = argparse.ArgumentParser()

def parse():
    sourceArgs(parser)
    args, rest = parser.parse_known_args(sys.argv)
    assert len(rest)==1, 'unknown parameters: '+str(rest[1:])
    return args

