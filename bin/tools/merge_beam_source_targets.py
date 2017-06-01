#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import codecs
import argparse


# hack for python2/3 compatibility
from io import open

argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Merge parallel beam predictions")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), required=True, nargs='+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.input = [codecs.open(f.name, encoding='UTF-8') for f in args.input]
    sources = [list(map(lambda x: x.replace("\n", ""), y.readlines())) for y in args.input]
    sources_len = [len(y) for y in sources]

    max_len = max(sources_len)
    len_factors = [max_len // len(y) for y in sources]

    flatten = lambda l: [item for sublist in l for item in sublist]

    sources = [flatten(list(map(lambda x: [x] * y_len, y))) 
                  for y, y_len in zip(sources, len_factors)]

    for lines in zip(*sources):
        print("\t".join(lines))

    args.input = [f.close() for f in args.input]

