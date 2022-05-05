import sys, os

import matplotlib.pyplot as plt

from dask import delayed

import cv2
from PIL import Image
from skimage.color import rgb2gray
import time as time
import ctypes
import multiprocessing as mp

import numpy as np
import scipy as sp
import pickle
from vj import BoostedCascade
from utils import integral_image

class FaceDetector:

    def __init__(self, modelpath, max_parallel_process=0):
        self.clf = BoostedCascade.loadModel(modelpath)
        self.detectWnd = (self.clf.detectWndW, self.clf.detectWndH)
        if max_parallel_process > 0:
            self.signal = mp.Value(ctypes.c_bool, False)
            self.image_queue = mp.Queue()
            self.result_queue = mp.Queue()
            self.setParallel(max_parallel_process)

    def stopParallel(self):
        if self.signal.value:
            self.signal.value = False
    
    def setParallel(self, max_parallel_process):
        if max_parallel_process <= 0:
            max_parallel_process = 1
        self.max_parallel_process = max_parallel_process
        if self.signal.value:
            self.signal.value = False
        self.signal = mp.Value(ctypes.c_bool, True)
        self.processes = [None] * max_parallel_process
        for tid in range(max_parallel_process):
            self.processes[tid] = mp.Process(target=__class__._parallel_detect, args=(self.signal, self.image_queue, self.result_queue, self.clf))
            self.processes[tid].start()
            
    def _parallel_detect(signal, image_queue, result_queue, clf):
        while signal.value:
            if image_queue.empty():
                time.sleep(0.2)
            else:
                data = image_queue.get()
                if type(data) != type(None):
                    pred = clf.predictIntegralImage(data[:, 0])
                    result_queue.put([len(data), data[pred == 1, 1:]])
                    
    def getImageTiles(ii, wndW, wndH, padX, padY):
        h, w = ii.shape
        data = []
        for y in range(0,h-wndH,padY):
            for x in range(0,w-wndW,padX):
                data.append( [ ii[y:y+wndH, x:x+wndW], x, y, wndW, wndH ] )
        return np.array(data, dtype=object)

    def detect(self, image, min_size=0.0, max_size=1.0, step=0.5, detectPad=(12, 12), verbose=False, getTotalTiles=False):
        while self.lock:
            continue
        self.lock = True
        faces = []
        height, width = image.shape

        if min_size * min(width, height) < self.detectWnd[0]:
            min_size = self.detectWnd[0] / min(width, height)
        if max_size < min_size:
            print("[WARNING] max_size < min_size")
            max_size = min_size
        assert step > 0.0 and step < 1.0

        total_items = 0
        given_items = 0
        done_items = 0

        si = max_size
        while True:
            scaledimg =  cv2.resize(image, (int(si*width), int(si*height)), interpolation = cv2.INTER_AREA)
            ii = integral_image(scaledimg)
            if scaledimg.shape[0]==0 or scaledimg.shape[1]==0:
                break
            data = __class__.getImageTiles(
                ii,
                self.detectWnd[0],
                self.detectWnd[1],
                detectPad[0],
                detectPad[1]
            )
            
            if data.shape[0]==0:
                break
            if verbose:
                print("Shape:", data.shape)
            n_samples, _ = np.shape(data)
            if verbose:
                print("tiles count:", n_samples, "tiles shape:", data[0,0].shape)
            total_items += n_samples

            for ind in range(n_samples):
                data[ind] = [
                    data[ind, 0],
                    int(data[ind, 1]/si),
                    int(data[ind, 2]/si),
                    int(data[ind, 3]/si),
                    int(data[ind, 4]/si)
                ]


            for ind in range(0, n_samples, 100):
                blockbegin = ind
                blockend = min(ind+100, n_samples)
                self.image_queue.put(data[blockbegin:blockend, :])
                given_items += blockend - blockbegin
            
            if si <= min_size:
                break
            si = si * step
            if si < min_size:
                si = min_size
        if verbose:
            print("Total tiles:", total_items)
        assert given_items == total_items
        
        while True:
            if self.result_queue.empty():
                print(' %d/%d' % (done_items, total_items), end='\r')
                if self.image_queue.empty() and self.result_queue.empty():
                    if done_items == total_items:
                        break
                time.sleep(0.2)
            else:
                numbers, scaledfaces = self.result_queue.get()
                done_items += numbers
                for x, y, w, h in scaledfaces:
                    faces.append([x, y, w, h])

        if getTotalTiles:
            self.lock = False
            return np.array(faces), total_items
        else:
            self.lock = False
            return np.array(faces)

    def detect_seq(self, image, min_size=0.0, max_size=1.0, step=0.5, detectPad=(12, 12), verbose=False, getTotalTiles=False):

        self.lock = True
        faces = []
        height, width = image.shape

        if min_size * min(width, height) < self.detectWnd[0]:
            min_size = self.detectWnd[0] / min(width, height)
        if max_size < min_size:
            print("[WARNING] max_size < min_size")
            max_size = min_size
        assert step > 0.0 and step < 1.0

        total_items = 0
        given_items = 0
        done_items = 0

        si = max_size
        while True:
            scaledimg =  cv2.resize(image, (int(si*width), int(si*height)), interpolation = cv2.INTER_AREA)
            ii = integral_image(scaledimg)
            if scaledimg.shape[0]==0 or scaledimg.shape[1]==0:
                break
            data = __class__.getImageTiles(
                ii,
                self.detectWnd[0],
                self.detectWnd[1],
                detectPad[0],
                detectPad[1]
            )
            
            if data.shape[0]==0:
                break
            if verbose:
                print("Shape:", data.shape)
            n_samples, _ = np.shape(data)
            if verbose:
                print("tiles count:", n_samples, "tiles shape:", data[0,0].shape)
            total_items += n_samples

            for ind in range(n_samples):
                data[ind] = [
                    data[ind, 0],
                    int(data[ind, 1]/si),
                    int(data[ind, 2]/si),
                    int(data[ind, 3]/si),
                    int(data[ind, 4]/si)
                ]
            pred = self.clf.predictIntegralImage(data[:, 0])
            numbers = len(data)
            scaledfaces = data[pred == 1, 1:]
            for x, y, w, h in scaledfaces:
                faces.append([x, y, w, h])
            
            if si <= min_size:
                break
            si = si * step
            if si < min_size:
                si = min_size
        if verbose:
            print("Total tiles:", total_items)

        if getTotalTiles:
            return np.array(faces), total_items
        else:
            return np.array(faces)