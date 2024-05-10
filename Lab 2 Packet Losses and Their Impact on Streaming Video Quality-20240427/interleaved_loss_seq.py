#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     interleaved_loss_seq.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2023-04-17
#
# @brief An example of interleaved loss sequence
#


import numpy as np
import os
import sys
np.set_printoptions(threshold=np.inf)

sys.path.insert(0, os.getcwd())  # "conv_interleave.py" is in the current directory
from conv_interleave import conv_interleave, conv_deinterleave, print_binary
import numpy as np
'''
从0-187，按顺序把数据分别放在11个寄存器里，这是第一个packet，从188-375，把上一个packet的数据取出
'''
x1 = np.concatenate([np.ones(2040), np.zeros(2244)]).astype(int)
# input symbols (ones) of 10 DVB packets followed by zeros to clear  x1:1代表要传输的信息,后面跟着一串的要相除的0
# the shift registers in the interleaver
d1 = [int(i) for i in "17,34,51,68,85,102,119,136,153,170,187".split(',')]
d2 = d1[::-1]
x2 = conv_interleave(x1, d1)  # convolved symbols
print(list(x2))
np.random.seed(2)
loss_seq1 = np.random.randint(2, size=len(x2))  # loss -> 1   用来表示传输过程哪里会发生loss
# print(loss_seq1)

y = x2 * loss_seq1  # symbols affected by losses   表示x2在传输过程发生loss
# 经过卷积交错后，x2的每个packet里面，因为只有1代表信息，
# 所以x2 * loss_seq1后，1分散在每个packet前和每个packet里面检错symbols里
z = conv_deinterleave(y, d2)
print(loss_seq1)
print(list(x2))
# print(list(loss_seq1))
loss_seq2 = z[2244:]  # remove prepended zeros (i.e., those from SRSs)
print_binary(loss_seq2, "Interleaved loss sequence")
print(len(loss_seq2))
test = list(loss_seq2)
print('how many 0:',test.count(0))