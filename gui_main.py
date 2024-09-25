# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:19:28 2024

@author: Kirko
"""
from PyQt5 import QtWidgets, QtGui, QtCore, uic
import PIL, io, random, sys
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import Praktikum_ML_train_mnist_cnn

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        uic.loadUi("drawArea_MNIST.ui", self)
        self.setWindowTitle("Draw your digit")
        self.pixmap = QtGui.QPixmap(560,560)
        self.pixmap.fill(QtCore.Qt.black)
        self.lb_drawArea.setPixmap(self.pixmap)      
        self.last_x, self.last_y = None, None
        self.praktikum = Praktikum_ML_train_mnist_cnn
        # self.addToolBar(NavigationToolbar(self.widget_mpl))
        self.start() 

    def start(self):
        self.pb_reset.clicked.connect(self.reset_drawArea)
 
    def reset_drawArea(self):
        self.pixmap.fill(QtCore.Qt.black)
        self.lb_drawArea.setPixmap(self.pixmap)
       
    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.lb_drawArea.pixmap())
        pen = QtGui.QPen(QtCore.Qt.white,25)
        painter.setPen(pen)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
        qimage = QtWidgets.QWidget.grab(self.lb_drawArea).scaled(28,28).toImage()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer().ReadWrite)
        qimage.save(buffer, "png")
        pilimg = PIL.Image.open(io.BytesIO(buffer.data())).convert("L")
        pilimg = np.asarray(pilimg)
        pilimg = np.expand_dims(pilimg, axis=0)
        # img_predict = model.predict(pilimg)
        img_predict = self.praktikum.plot_from_gui(pilimg)
        print(img_predict)
        self.update_graph(img_predict)
        for i, x in enumerate(img_predict[0]):
            print(i, "=", x*100)       
        # plot_predict("?")
        
    def update_graph(self, img_predict):        
        self.widget_mpl.plot_predict_in_canvas(img_predict)  
        
        
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()   
window.show()    
app.exec()