#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     dfr_simulation.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-05-15
#           2023-04-11
#           2024-04-25
#
# @brief Skeleton code for the simulation of video streaming to investigate the
#        impact of packet losses on the quality of video streaming based on
#        decodable frame rate (DFR)
#


import argparse
import math
import numpy as np
import os
import sys
import random
from random import randint

sys.path.insert(0, os.getcwd())  # assume that your own modules in the current directory
from conv_interleave import conv_interleave, conv_deinterleave
from sgm_generate import sgm_generate


def dfr_simulation(
        random_seed,
        num_frames,
        loss_probability,
        video_trace,
        fec,
        ci):
    np.random.seed(random_seed)  # set random seed

    # N.B.: Obtain the information of the whole frames to create a loss
    # sequence in advance due to the optional convolutional
    # interleaving/deinterleaving.
    with open(video_trace, "r") as f:
        lines = f.readlines()[1:num_frames + 1]  # the first line is a comment.

    f_number = np.empty(num_frames, dtype=np.uint)
    f_type = [''] * num_frames
    f_pkts = np.empty(num_frames, dtype=np.uint)  # the number of packets per frame
    for i in range(num_frames):
        f_info = lines[i].split()
        f_number[i] = int(f_info[0])  # str -> int
        f_type[i] = f_info[2]
        f_pkts[i] = math.ceil(int(f_info[3]) / (188 * 8))

    # symbol loss sequence
    p = 1e-4
    q = p * (1.0 - loss_probability) / loss_probability
    n_pkts = sum(f_pkts)  # the number of packets for the whole frames



    # test = list(loss_seq2)
    # print('len:', test.count(0))

    # print(loss_seq2[10000:50000])

    # print('how many 0:', test.count(0))
    # initialize variables.
    idx = -1
    for j in range(2):
        idx = f_type.index('I', idx + 1)
    gop_size = f_number[idx]  # N.B.: the frame number of the 2nd I frame is GOP size.
    num_b_frames = f_number[1] - f_number[0] - 1  # between I and the 1st P frames
    i_frame_number = -1  # the last decodable I frame number
    p_frame_number = -1  # the last decodable P frame number
    # num_frames_decoded = 0
    num_pkts_received = 0
    num_frames_decoded = 0
    num_frames_received = 0

    frame_loss = False
    I_loss, P_loss = True, True
    B1_frame_decodeable, B2_frame_decodeable, B3_frame_decodeable = True, True, True
    # main loop
    for i in range(num_frames):

        # frame loss
        # pkt_losses = sum(loss_seq2[num_pkts_received:num_pkts_received + f_pkts[i]])
        num_pkts_received += f_pkts[i]
        num_frames_received += 1
        if ci:
            # apply convolutional interleaving/deinterleaving.
            # N.B.:
            # 1. Append 2244 zeros before interleaving.
            # 2. Interleaved sequence experiences symbol losses.
            # 3. Remove leading 2244 elements after deinterleaving.
            x1 = np.concatenate([np.ones(204 * f_pkts[i]), np.zeros(2244)])
            d1 = [int(i) for i in "17,34,51,68,85,102,119,136,153,170,187".split(',')]
            d2 = d1[::-1]
            x2 = conv_interleave(x1, d1)  # convolved symbols
            loss_seq1 = sgm_generate(777, len(x2), p, q)
            # loss_seq1 = [loss_seq1[i] ^ 1 for i in loss_seq1]
            # loss_seq1 = np.random.randint(2, size=len(x2))
            y = x2 * loss_seq1  # symbols affected by losses
            z = conv_deinterleave(y, d2)
            loss_seq2 = z[2244:]  # remove prepended zeros (i.e., those from SRSs)
            # print('loss_seq2:', loss_seq2)

            # TODO: Implement.
        else:  # not convolutional interleaving/deinterleaving (CI)
            x1 = np.ones(188 * f_pkts)
            loss_seq1 = sgm_generate(777, len(x1), p, q)
            # loss_seq1 = [loss_seq1[i] ^ 1 for i in loss_seq1]
            # print(loss_seq1)
            # test = list(loss_seq1)
            # print('len:', test.count(1))
            loss_seq2 = x1 * loss_seq1  # symbols affected by losses
            # TODO: Implement.

        if fec:
            total_pkt = f_pkts[i]

            # to see each pkt in ith frame, and see if any loss in current pkt
            for k in range(0, total_pkt):  # total_pkt - 1, -1, -1
                pkt = loss_seq2[(num_pkts_received - k - 1) * 204: (num_pkts_received - k) * 204]
                pkt_list = list(pkt)
                # if current pkt for current frame has more than 8 symbols loss
                # then it means that current pkt is loss and so that current ith frame is loss
                # and do not need to see the rest of pkt of ith frame
                if pkt_list.count(1) > 8:
                    frame_loss = True  # frame is loss
                    break
                else:
                    frame_loss = False

        # TODO: Set "frame_loss" based on "pkt_losses" with FEC.

        else:
            total_pkt = f_pkts[i]
            for k in range(0, total_pkt):  # total_pkt - 1, -1, -1
                pkt = loss_seq2[(num_pkts_received - k - 1) * 188: (num_pkts_received - k) * 188]
                pkt_list = list(pkt)
                # if current pkt for current frame has more than 1 symbols loss
                # then it means that current pkt is loss and so that current ith frame is loss
                # and do not need to see the rest of pkt of ith frame
                if pkt_list.count(1) >= 1:
                    frame_loss = True  # frame is loss
                    break
                else:
                    frame_loss = False
            # TODO: Set "frame_loss" based on "pkt_losses" without FEC.

        # frame decodability
        if not frame_loss:  # see the fec-dependent handling of "frame_loss" above.
            # print('before')
            # print('i:', i, 'I_loss', I_loss, 'i_frame_number', i_frame_number, 'P_loss', P_loss, p_frame_number)
            # print('B1_frame_decodeable', B1_frame_decodeable, "B2_frame_decodeable", B2_frame_decodeable,
            #       "B3_frame_decodeable", B3_frame_decodeable, 'num_frames_decoded', num_frames_decoded,'\n')
            match f_type[i]:
                case 'I':
                    # print('I', i, 'i_frame_number', i_frame_number)
                    num_frames_decoded += 1
                    # print('I decode')

                    if B1_frame_decodeable == False and I_loss == False:
                        num_frames_decoded += 1
                        # print('B decode')
                        B1_frame_decodeable = False
                    if B2_frame_decodeable == False and I_loss == False:
                        num_frames_decoded += 1
                        # print('B decode')
                        B2_frame_decodeable = False
                    if B3_frame_decodeable == False and I_loss == False:
                        num_frames_decoded += 1
                        # print('B decode')
                        B3_frame_decodeable = False
                    # print('\n')
                    # TODO: Implement.

                case 'P':
                    # 'IBBBP' or 'PBBBP' or 'IPBBB'
                    # print('P', 'i - 1:', i - 1, 'i - 4:', i - 4, 'i_frame_number:', i_frame_number, "p_frame_number:",
                    #       p_frame_number)
                    if i - 4 == i_frame_number or i - 4 == p_frame_number or i - 1 == i_frame_number:
                        num_frames_decoded += 1
                        # print('P decode ')

                    if B1_frame_decodeable == False and P_loss == False:
                        num_frames_decoded += 1
                        # print('B decode ')
                        B1_frame_decodeable = True
                    if B2_frame_decodeable == False and P_loss == False:
                        num_frames_decoded += 1
                        # print('B decode ')
                        B2_frame_decodeable = True
                    if B3_frame_decodeable == False and P_loss == False:
                        num_frames_decoded += 1
                        # print('B decode ')
                        B3_frame_decodeable = True
                    # print('\n')
                    # TODO: Implement.

                case 'B':
                    # for the first B frame in case 'IPBBB'
                    # print('B:', 'i:', i, 'i_frame_number', i_frame_number, 'p_frame_number', p_frame_number)
                    if i - 2 == i_frame_number and i - 1 == p_frame_number:
                        # print('B: now1')
                        num_frames_decoded += 1
                        continue

                    # for the first B frame in case 'IPBBB'
                    if i - 3 == i_frame_number and i - 2 == p_frame_number:
                        # print('B:now2')
                        num_frames_decoded += 1
                        continue

                    # for the first B frame in case 'IPBBB'
                    if i - 4 == i_frame_number and i - 3 == p_frame_number:
                        # print('B:now3')
                        num_frames_decoded += 1
                        continue

                    # for the first B frame in case 'IBBBP' or 'PBBBP' or 'PBBBI'
                    if i - 1 == i_frame_number or i - 1 == p_frame_number:
                        B1_frame_decodeable = False
                        # print("B1_frame_decodeable", B1_frame_decodeable)
                        continue
                    # for the second B frame in case 'IBBBP'
                    if i - 2 == i_frame_number or i - 2 == p_frame_number:
                        B2_frame_decodeable = False
                        # print("B2_frame_decodeable", B2_frame_decodeable)
                        continue
                    # for the third B frame in case 'IBBBP'
                    if i - 3 == i_frame_number or i - 3 == p_frame_number:
                        B3_frame_decodeable = False
                        # print("B3_frame_decodeable", B3_frame_decodeable)
                        continue

                    # print('\n')
                    # TODO: Implement.

                case _:
                    sys.exit("Unkown frame type is detected.")

        if f_type[i] == 'I':
            if frame_loss:
                I_loss = True
            else:
                I_loss = False
                # update:the index of the last decodable I frame,
                # it means current I frame is not loss and decodable
                i_frame_number = i
        if f_type[i] == 'P':
            if frame_loss:
                P_loss = True
            else:
                P_loss = False
                # update:the index of the last decodable P frame,
                # it means current P frame is not loss and decodable
                p_frame_number = i
    #     print('after')
    #     print('i:', i, 'I_loss', I_loss, 'i_frame_number', i_frame_number, 'P_loss', P_loss, p_frame_number)
    #     print('B1_frame_decodeable', B1_frame_decodeable, "B2_frame_decodeable", B2_frame_decodeable,
    #           "B3_frame_decodeable", B3_frame_decodeable, 'num_frames_decoded', num_frames_decoded,'\n')
    # print(num_frames_decoded, num_frames)
    return num_frames_decoded / num_frames  # DFR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--num_frames",
        help="number of frames to simulate; default is 10000",
        default=10000,  # 10000
        type=int)
    parser.add_argument(
        "-P",
        "--loss_probability",
        help="overall loss probability; default is 0.1",
        default=1e-4,
        type=float)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="seed for numpy random number generation; default is 777",
        default=777,
        type=int)
    parser.add_argument(
        "-V",
        "--video_trace",
        help="video trace file; default is 'silenceOfTheLambs_verbose'",
        default="silenceOfTheLambs_verbose",
        type=str)

    # convolutional interleaving/deinterleaving (CI); default is False
    parser.add_argument('--ci', dest='ci', action='store_true')
    parser.add_argument('--no-ci', dest='ci', action='store_false')
    parser.set_defaults(ci=False)

    # forward error correction (FEC); default is False (i.e., not using FEC)
    parser.add_argument('--fec', dest='fec', action='store_true')
    parser.add_argument('--no-fec', dest='fec', action='store_false')
    parser.set_defaults(fec=True)

    args = parser.parse_args()

    # set variables using command-line arguments
    num_frames = args.num_frames
    # loss_model = args.loss_model
    loss_probability = args.loss_probability
    video_trace = args.video_trace
    ci = args.ci
    fec = args.fec
    # trace = args.trace

    # run simulation and display the resulting DFR.
    dfr = dfr_simulation(
        args.random_seed,
        args.num_frames,
        args.loss_probability,
        args.video_trace,
        args.fec,
        args.ci)
    print(f"Decodable frame rate = {dfr:.4E}\n")
