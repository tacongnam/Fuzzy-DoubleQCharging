import numpy as np

def get_location(mc):
    dx = mc.end[0] - mc.start[0]
    dy = mc.end[1] - mc.start[1]
    d = np.hypot(dx, dy)
    time_move = d / mc.velocity

    if time_move == 0:
        return mc.current
    elif np.hypot(mc.current[0] - mc.end[0], mc.current[1] - mc.end[1]) < 1e-3:
        return mc.end
    else:
        # Move 2 seconds
        x_hat = mc.current[0] + 2 * dx / time_move
        y_hat = mc.current[1] + 2 * dy / time_move

        if (dx * (mc.end[0] - x_hat) < 0 or
            (dx * (mc.end[0] - x_hat) == 0 and dy * (mc.end[1] - y_hat) <= 0)):
            return mc.end
        else:
            return x_hat, y_hat

def charging(mc, net, node):
    for nd in net.node:
        p = nd.charge(mc)
        mc.energy -= p
