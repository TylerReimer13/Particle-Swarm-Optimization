import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


def f(x, y):
    f = x ** 2 + (y + 1) ** 2 - 5 * np.cos(1.5 * x + 1.5) - 3 * np.cos(2 * x - 1.5)
    # f = -200 * np.exp(-.2 * np.sqrt(x ** 2 + y ** 2))
    # f = np.cos(x) * np.sin(y) - (x / (y ** 2 + 1))
    # f = (x + 10) ** 2 + (y + 10) ** 2 + np.exp(-x ** 2 - y ** 2)
    return f


class Particle:
    def __init__(self, pos, vel):
        self.p = pos
        self.z = f(self.p[0], self.p[1])

        self.v = vel

        self.pb_xy = self.p
        self.pb_z = self.z

        self.w = .8
        self.c1 = 2.
        self.c2 = 2.

    def update_coefs(self, itr, max_itr):
        self.w = .4 * ((itr - max_itr) / (max_itr ** 2)) + .4
        self.c1 = -3 * (itr / max_itr) + 3.5
        self.c2 = 3 * (itr / max_itr) + .5

    def update(self, swarm_best, r1, r2, i, ii):
        v_new = self.w * self.v + self.c1 * r1 * (self.pb_xy - self.p) + self.c2 * r2 * (swarm_best - self.p)

        self.p += v_new
        self.z = f(self.p[0], self.p[1])
        if self.z < self.pb_z:
            self.pb_xy = self.p
            self.pb_z = self.z

        self.update_coefs(i, ii)


class PSO:
    def __init__(self, n_particles, n_iter, x_bds, y_bds):
        self.n_particles = n_particles
        self.n_iter = n_iter

        self.x_bds = x_bds
        self.y_bds = y_bds

        random_x = np.random.uniform(self.x_bds[0], self.x_bds[1], self.n_particles)
        random_y = np.random.uniform(self.y_bds[0], self.y_bds[1], self.n_particles)
        random_v = (np.random.random((self.n_particles, 2)) - 0.5) / 10

        self.particles = []
        for n in range(self.n_particles):
            rand_pos = np.array([random_x[n], random_y[n]])
            rand_vel = np.array([random_v[n][0], random_v[n][1]])
            new_particle = Particle(rand_pos, rand_vel)
            self.particles.append(new_particle)

        self.swarm_best = np.zeros(2)
        self.find_swarm_best()

    def __call__(self):
        for i in range(self.n_iter):
            r1 = np.random.uniform(0., 1., self.n_particles)
            r2 = np.random.uniform(0., 1., self.n_particles)
            self.find_swarm_best()
            for i, ptc in enumerate(self.particles):
                ptc.update(self.swarm_best, r1[i], r2[i], i, self.n_iter)
        est = np.mean([pc.z for pc in self.particles])
        print('MIN ESTIMATE: ', est)

    def find_swarm_best(self):
        best = sorted(self.particles, key=lambda x: x.pb_z, reverse=False)[0]
        self.swarm_best = np.array([best.p[0], best.p[1]])

    def show_f(self):
        xr = np.linspace(self.x_bds[0], self.x_bds[1], 100)
        yr = np.linspace(self.y_bds[0], self.y_bds[1], 100)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(xr, yr)
        Z = f(X, Y)

        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False, alpha=.3)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter3D([pc.p[0] for pc in self.particles], [pc.p[1] for pc in self.particles],
                     [pc.z for pc in self.particles], s=2., c='k', alpha=1.)
        plt.show()


if __name__ == "__main__":
    pso = PSO(50, 100, [-5., 5.], [-5., 5.])
    pso.show_f()
    pso()
    pso.show_f()

