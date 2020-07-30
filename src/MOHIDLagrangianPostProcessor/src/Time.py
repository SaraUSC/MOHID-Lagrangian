#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:54:52 2019

@author: gfnl143
"""
import xml.etree.ElementTree as ET
import numpy as np
import MDateTime
from src.constants import daysToSeconds


class FilesTimesHandler:

    def __init__(self, fileList):
        self.fileList = fileList
        self.ISO_time_origin = MDateTime.getDateStringFromDateTime(MDateTime.BaseDateTime())
        self.timeAxis = []
        self.timeMask = []
        self.startTime = []
        self.endTime = []
        self.dt = []

    def getTimeAxisFromXML(self, xmlFile):
        root = ET.parse(xmlFile).getroot()
        for parameter in root.findall('execution/parameters/parameter'):
            if parameter.get('key') == 'Start':
                self.startTime = parameter.get('value')
            if parameter.get('key') == 'End':
                self.endTime = parameter.get('value')
            if parameter.get('key') == 'OutputWriteTime':
                self.dt = np.float(parameter.get('value'))
        nFiles = len(self.fileList)
        startTimeStamp = MDateTime.getTimeStampFromISODateString(self.startTime)
        self.timeAxis = np.array([startTimeStamp + i*self.dt/daysToSeconds for i in range(0,nFiles)])

    def getTimeMaskFromRecipe(self, xmlRecipe):
        root = ET.parse(xmlRecipe).getroot()
        self.timeMask = np.ones(self.timeAxis.size, dtype=np.bool)
        for parameter in root.findall('time/'):
            if parameter.tag == 'start':
                timeRecipeStart = MDateTime.getTimeStampFromISODateString(parameter.get('value'))
                self.timeMask = self.timeMask & (self.timeAxis > timeRecipeStart)
            if parameter.tag == 'end':
                timeRecipeEnd = MDateTime.getTimeStampFromISODateString(parameter.get('value'))
                self.timeMask = self.timeMask & (self.timeAxis < timeRecipeEnd)
            if parameter.tag == 'step':
                step = np.int64(parameter.get('value'))
                bufferTimeMask = np.zeros(self.timeAxis.size, dtype=np.bool)
                bufferTimeMask[::step] = True
                self.timeMask = self.timeMask & bufferTimeMask
        self.timeAxis = self.timeAxis[self.timeMask]

    def initializeTimeGrid(self, xmlFile, xmlRecipe):
        self.getTimeAxisFromXML(xmlFile)
        self.getTimeMaskFromRecipe(xmlRecipe)

    def cropFileList(self):
        tidx = 0
        croppedFileList = []
        for fileName in self.fileList:
            if self.timeMask[tidx] == True:
                croppedFileList.append(fileName)
            tidx += 1

        self.fileList = croppedFileList
        return croppedFileList