#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     conv_interleave.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2023-04-11
#
# @brief Functions for convolutional interleaving and deinterleaving.
#


import argparse
import collections
import numpy as np

np.set_printoptions(threshold=np.inf)
def conv_interleave(x, delays):
    """
    Interleave convolutionally the input sequence with the given delays.
    The first branch is assumed to have zero delay.
    从1开始一直到2之前（指1到2之间的‘0’），为一个packet（204 symbols），其中‘1’和‘2’用来指示当前为第几个packet。
    在这一个packet里，有17（不固定，按照17,34,51,68,85,102,119,136,153,170,187)个标志,和188个信息symbols。
    总共有11（1-11）个packet
    Args:
        x (numpy array): input symbols
        delays (list): branch delays
    """

    # initialize shift registers (SRS) based on deque
    srs = [] # list of SRSs
    for i in range(len(delays)):
        srs.append(collections.deque(delays[i]*[0], delays[i]))  # 长度为17,34,51,68,85,102,119,136,153,170,187的deque

    nb = len(delays) + 1 # number of branches 12
    cb = 0               # current branch
    y = np.empty_like(x)

    for i in range(len(x)):
        if i % nb == 0:
            y[i] = x[i]

        else:
            y[i] = srs[cb-1].pop() # N.B.: the index of SRs should be cb-1!
            srs[cb-1].appendleft(x[i])
        cb = (cb + 1) % nb

    return y


def conv_deinterleave(x, delays):
    """
    Deinterleave convolutionally the input sequence with the given delays.
    The last branch is assumed to have zero delay.
    去交错：在卷积交错后，我们得到11个packets,共11x204=2244个symbols，
    其中每个packet前面还有诸如‘1’，‘2’一直到‘11’的符号用来指示当前为第几个packets，
    在每个packet里包含（17,34,51,68,85,102,119,136,153,170,187）个标志和204-（）个信息symbols，即‘0’s。
    总共有2254个symbols，减去
    Args:
        x (numpy array): input symbols
        delays (list): branch delays
    """

    # initialize shift registers (based on deque)
    srs = []
    for i in range(len(delays)):
        srs.append(collections.deque(delays[i]*[0], delays[i]))

    nb = len(delays) + 1 # number of branches  12
    cb = 0               # current branch
    y = np.empty_like(x)
    # print(srs)
    for i in range(len(x)):
        if i % nb == nb-1:
            y[i] = x[i]

        else:
            y[i] = srs[cb].pop() # N.B.: the index of SRs should be cb this time!
            srs[cb].appendleft(x[i])

        cb = (cb + 1) % nb

    return y


def print_binary(x, desc=None):
    """Pretty-print binary sequence

    Args:
        x (iterable or numpy array): binary sequence
        desc (str): description
    """
    if desc is None:
        print(", ".join(map(str, x)))
    else:
        print(desc + ": " + ", ".join(map(str, x)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        help="number of input symbols; default is 2254 (10 symbols after interleaving/deinterleaving)",
        default=2254,
        type=int)
    parser.add_argument(
        "-d",
        help="delays in comma-separated string; default is '17,34,51,68,85,102,119,136,153,170,187' (i.e., DVB standard)",
        default="17,34,51,68,85,102,119,136,153,170,187",
        type=str)

    args = parser.parse_args()
    x = np.asarray([int(i) for i in range(1, args.n+1)])
    d1 = [int(i) for i in args.d.split(',')]
    d2 = d1[::-1]               # d1 reversed

    y = conv_interleave(x, d1)
    # np.set_printoptions(threshold=np.inf)
    # print(y)
    z = conv_deinterleave(y, d2)

    
    print_binary(x, "Original")
    print_binary(y, "Interleaved")
    print_binary(z, "Deinterleaved")
