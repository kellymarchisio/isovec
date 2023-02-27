###############################################################################
# This file is adapted slightly from euc-v-graph BLI by Kelly Marchisio.
#   https://github.com/kellymarchisio/euc-v-graph-bli/blob/main/dicts/make_devsets.py
# 
# Adapted by Kelly Marchisio, 2022.

import argparse
import sys
START_N=6501
END_N=8000

def make_dev(infile, outfile):
    with open(outfile, 'w') as o:
        uniq_wdcount = 0
        last_word = None
        for line in open(infile, 'r'):
            srcwd, trgwd = line.split()
            if srcwd == last_word and uniq_wdcount >= START_N:
                o.write(line)
            elif srcwd != last_word:
                last_word = srcwd
                uniq_wdcount += 1
                if uniq_wdcount > END_N:
                    print('Dev creation complete. Exiting.')
                    o.close()
                    return
                elif uniq_wdcount >= START_N:
                    o.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='source language abbreviation')
    parser.add_argument('trg', help='source language abbreviation')
    args = parser.parse_args()
    lang_pair = args.src + '-' + args.trg

    infile = lang_pair + '/' + lang_pair + '.txt'
    outfile = (lang_pair + '/dev/' + lang_pair + '.' + str(START_N) + '-' +
            str(END_N) + '.txt')
    make_dev(infile, outfile)
