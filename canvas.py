# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:36:19 2024

@author: Admin
"""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }

class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = plt.Figure() 
        super().__init__(self.figure)
        # self.toolbar = NavigationToolbar(self.figure, self)
        self.ax = self.figure.subplots()  
        self.ax.set(xticks=np.arange(-1,11,1))  
        xlabels = self.ax.get_xticklabels()
        xlabels[0].set_visible(False)
        xlabels[-1].set_visible(False) 
        self.ax.grid()       
               
    def plot_predict_in_canvas(self, img_predict):
        self.ax.cla()       
        self.ax.bar(range(10), img_predict[0], color="blue")
        self.ax.set(xticks=np.arange(-1,11,1))  
        xlabels = self.ax.get_xticklabels()
        xlabels[0].set_visible(False)
        xlabels[-1].set_visible(False)
        self.ax.set_xlabel('class', fontdict=font)
        self.ax.set_ylabel('confidence', fontdict=font)
        self.ax.set_title('Classification', fontdict=font)
        self.ax.grid()
        self.draw()
