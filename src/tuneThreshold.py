#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb
import argparse

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), numpy.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr)

def tuneThresholdfromScore_std(scores, labels, target_fa=None, target_fr=None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    ## compute minc@0.01 and minc@0.001
    minc_s = compute_c_norm(fnr, fpr, p_target=0.01)
    minc_ss = compute_c_norm(fnr, fpr, p_target=0.001)
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    if target_fa:
        for tfa in target_fa:
            idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr, minc_s, minc_ss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Scoring")

    parser.add_argument('--trial_outs', type=str, help='trial outputs')
    parser.add_argument('--trial_keys', type=str, help='trial keys')

    args = parser.parse_args()

    score_dict = {}
    with open(args.trial_outs, 'r') as f:
        for score_count, line in enumerate(f):
            score, enroll, test = line[:-1].split(' ')
            score_dict[enroll+'.'+test] = float(score)

    key_dict = {}
    with open(args.trial_keys, 'r') as f:
        for key_count, line in enumerate(f):
            key, enroll, test = line[:-1].split(' ')
            key_dict[enroll+'.'+test] = int(key)
    
    assert score_count == key_count

    scores = []
    keys = []

    for i in key_dict:
        keys.append(key_dict[i])
        scores.append(score_dict[i])
    
    results = tuneThresholdfromScore_std(scores, keys, target_fa=None, target_fr=None)

    print('###'+args.trial_outs)
    print('EER: '+str(results[1]))
    print('MINC@0.01: '+str(results[-2]))
    print('MINC@0.001: '+str(results[-1]))
    