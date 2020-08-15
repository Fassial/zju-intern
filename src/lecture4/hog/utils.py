"""
Created on July 22 01:38, 2020

@author: fassial
"""

"""
line:
    get the coordinate point between the two points
    @params:
        r0(int) : the row-value(y) of the first point
        c0(int) : the col-value(x) of the first point
        r1(int) : the row-value(y) of the second point
        c1(int) : the col-value(x) of the second point
    @rets:
        res_y(list) : the row-value(y) of the coordinate point between the two points
        res_x(list) : the col-value(x) of the coordinate point between the two points
"""
def line(r0, c0, r1, c1, n_min = 5):
    # get cycle & step
    cycle = n_min if (r1 - r0) <= n_min else abs(r1 - r0)
    step_y, step_x = (r1 - r0) / cycle, (c1 - c0) / cycle
    cycle += 1
    # init res_y & res_x
    res_y, res_x = [], []
    # init start y, start x
    y, x = r0, c0
    for _ in range(cycle):
        res_y.append(round(y)), res_x.append(round(x))
        x += step_x
        y += step_y
    return (res_y, res_x)
