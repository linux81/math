import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry
g = 9.81
L1 = 1.0
L2 = 1.0
m1 = 1.0
m2 = 1.0

# Równania ruchu
def deriv(state):
    th1, w1, th2, w2 = state
    d = th2 - th1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(d)**2
    den2 = (L2 / L1) * den1

    dw1 = (m2*L1*w1**2*np.sin(d)*np.cos(d) +
           m2*g*np.sin(th2)*np.cos(d) +
           m2*L2*w2**2*np.sin(d) -
           (m1 + m2)*g*np.sin(th1)) / den1
    dw2 = (-m2*L2*w2**2*np.sin(d)*np.cos(d) +
           (m1 + m2)*(g*np.sin(th1)*np.cos(d) -
                      L1*w1**2*np.sin(d) -
                      g*np.sin(th2))) / den2
    return np.array([w1, dw1, w2, dw2])

def rk4_step(state, dt):
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Warunki początkowe (prawie identyczne)
state1 = np.array([np.pi/2, 0.0, np.pi/2 - 0.01, 0.0])
state2 = state1.copy()
state2[2] += 0.001  # subtelna różnica

dt = 0.01
steps = 5000

# Animacja
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_title('Wahadło podwójne: wrażliwość na warunki początkowe')

line1, = ax.plot([], [], 'b-', lw=2, alpha=0.7)
line2, = ax.plot([], [], 'r-', lw=2, alpha=0.7)
trace1, = ax.plot([], [], 'b.', ms=1, alpha=0.3)
trace2, = ax.plot([], [], 'r.', ms=1, alpha=0.3)

x1t, y1t, x2t, y2t = [], [], [], []

def get_points(state):
    th1, w1, th2, w2 = state
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    return x1, y1, x2, y2

def update(frame):
    global state1, state2
    state1 = rk4_step(state1, dt)
    state2 = rk4_step(state2, dt)

    x1a, y1a, x2a, y2a = get_points(state1)
    x1b, y1b, x2b, y2b = get_points(state2)

    line1.set_data([0, x1a, x2a], [0, y1a, y2a])
    line2.set_data([0, x1b, x2b], [0, y1b, y2b])

    x1t.append(x2a); y1t.append(y2a)
    x2t.append(x2b); y2t.append(y2b)
    trace1.set_data(x1t[-1500:], y1t[-1500:])
    trace2.set_data(x2t[-1500:], y2t[-1500:])

    return [line1, line2, trace1, trace2]

ani = animation.FuncAnimation(fig, update, frames=steps, interval=10, blit=True)
plt.show()
