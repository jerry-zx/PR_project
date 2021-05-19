from graphics import *
import numpy as np


def predict_observe(w_1, w_2, b, width, test_num, win):
    for i in range(test_num):
        p = win.getMouse()
        c = Circle(p, width / 100)
        x_1, x_2 = p.getX(), p.getY()
        if w_1 * x_1 + w_2 * x_2 + b > 0:
            c.setFill("red")
            c.setOutline("red")
        else:
            c.setFill("blue")
            c.setOutline("blue")
        c.draw(win)
