import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import eval_legendre


def p_set_6_no_1b():
    resolution: int = 1_000
    iterations: int = 100_000
    x_values = np.linspace(-1.0, 1.0, resolution)
    y_values = np.linspace(-1.0, 1.0, resolution)

    @njit
    def potential_summation_term(x: float, y: float, m: int):
        return 4*(np.exp(m*np.pi*(y-1))-np.exp(-m*np.pi*(y+1)))/(1-np.exp(-2*m*np.pi)) * np.sin(m*np.pi*x)/(m*np.pi)

    @njit
    def potential_constant_x(x: float, y_points: np.ndarray = y_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(y_points)
        sum_term = np.empty_like(y_points)
        for m in (2*integer_index+1):
            for i, y in enumerate(y_points):
                sum_term[i] = potential_summation_term(x, y, m)
            output += sum_term
        return output

    @njit
    def potential_constant_y(y: float, x_points: np.ndarray = y_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(x_points)
        sum_term = np.empty_like(x_points)
        for m in (2 * integer_index + 1):
            for i, x in enumerate(x_points):
                sum_term[i] = potential_summation_term(x, y, m)
            output += sum_term
        return output

    def initialize_x_plot():
        plt.xlabel("$x[a]$")
        plt.ylabel(r"$\Phi(x)[V_0]$")
        plt.legend(loc="upper left")

    def initialize_y_plot():
        plt.xlabel("$y[b]$")
        plt.ylabel(r"$\Phi(y)[V_0]$")
        plt.legend(loc="upper left")

    for i in (0.01, 0.1, 0.5):
        plt.plot(x_values, potential_constant_y(i), label=f"$y={i}b$")
    initialize_x_plot()
    plt.savefig("const_y_1.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    for i in (0.9, 0.99, 1.0):
        plt.plot(x_values, potential_constant_y(i), label=f"$y={i}b$")
    initialize_x_plot()
    plt.savefig("const_y_2.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    for i in (0.01, 0.1, 0.5):
        plt.plot(y_values, potential_constant_x(i), label=f"$x={i}a$")
    initialize_y_plot()
    plt.savefig("const_x_1.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    for i in (0.9, 0.99, 1.0):
        plt.plot(y_values, potential_constant_x(i), label=f"$x={i}a$")
    initialize_y_plot()
    plt.savefig("const_x_2.pdf", bbox_inches="tight")
    plt.legend()
    print("done")


def p_set_6_no_1d():
    resolution: int = 1_000
    iterations: int = 100_000
    x_values = np.linspace(-1.0, 1.0, resolution)
    y_values = np.linspace(-1.0, 1.0, resolution)

    @njit
    def Ex_summation_term(x: float, y: float, m: int):
        return -4*(np.exp(m*np.pi*(y-1))-np.exp(-m*np.pi*(y+1)))/(1-np.exp(-2*m*np.pi)) * np.cos(m*np.pi*x)

    @njit
    def Ey_summation_term(x: float, y: float, m: int):
        return -4*(np.exp(m*np.pi*(y-1))+np.exp(-m*np.pi*(y+1)))/(1-np.exp(-2*m*np.pi)) * np.sin(m*np.pi*x)

    @njit
    def Ex_constant_x(x: float, y_points: np.ndarray = y_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(y_points)
        sum_term = np.empty_like(y_points)
        for m in (2*integer_index+1):
            for i, y in enumerate(y_points):
                sum_term[i] = Ex_summation_term(x, y, m)
            output += sum_term
        return output

    @njit
    def Ex_constant_y(y: float, x_points: np.ndarray = x_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(x_points)
        sum_term = np.empty_like(x_points)
        for m in (2*integer_index+1):
            for i, x in enumerate(x_points):
                sum_term[i] = Ex_summation_term(x, y, m)
            output += sum_term
        return output

    @njit
    def Ey_constant_x(x: float, y_points: np.ndarray = y_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(y_points)
        sum_term = np.empty_like(y_points)
        for m in (2*integer_index+1):
            for i, y in enumerate(y_points):
                sum_term[i] = Ey_summation_term(x, y, m)
            output += sum_term
        return output

    @njit
    def Ey_constant_y(y: float, x_points: np.ndarray = x_values, N: int = iterations):
        integer_index = np.arange(N)
        output = np.zeros_like(x_points)
        sum_term = np.empty_like(x_points)
        for m in (2*integer_index+1):
            for i, x in enumerate(x_points):
                sum_term[i] = Ey_summation_term(x, y, m)
            output += sum_term
        return output

    def initialize_Ex_x_plot():
        plt.xlabel("x[a]")
        plt.ylabel(r"$E_x(x)[V_0/a]$")
        plt.legend()

    def initialize_Ex_y_plot():
        plt.xlabel("y[b]")
        plt.ylabel(r"$E_x(y)[V_0/a]$")
        plt.legend()

    def initialize_Ey_x_plot():
        plt.xlabel("x[a]")
        plt.ylabel(r"$E_y(x)[V_0/b]$")
        plt.legend()

    def initialize_Ey_y_plot():
        plt.xlabel("y[b]")
        plt.ylabel(r"$E_y(y)[V_0/b]$")
        plt.legend()

    plt.plot(y_values, Ex_constant_x(0.5), label=f"$x={0.5}a$")
    initialize_Ex_y_plot()
    plt.savefig("Ex_const_x.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    for i in (0, 0.5, 0.99):
        plt.plot(x_values, Ex_constant_y(i), label=f"$y={i}b$")
    initialize_Ex_x_plot()
    plt.savefig("Ex_const_y.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    for i in (0, 0.5, 0.99):
        plt.plot(x_values, Ex_constant_y(i), label=f"$y={i}b$")
    initialize_Ex_x_plot()
    plt.ylim((-1, 1))
    plt.savefig("Ex_const_y_zoom.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    plt.plot(x_values, Ey_constant_y(0), label=f"$y={0}b$")
    initialize_Ey_x_plot()
    plt.savefig("Ey_const_y.pdf", bbox_inches="tight")
    plt.clf()
    print("done")

    plt.plot(y_values, Ey_constant_x(0.5), label=f"$x={0.5}a$")
    initialize_Ey_y_plot()
    plt.savefig("Ey_const_x.pdf", bbox_inches="tight")
    plt.clf()
    print("done")


def p_set_6_no_3c():
    resolution: int = 1_000
    iterations: int = 500
    x_values = np.linspace(0, np.pi, resolution)
    cos_values = np.cos(x_values)

    def out_summation_term(k: int, cos):
        return (-1)**k*eval_legendre(2*k, cos)/((2*k+3)*(2*k+1))

    def out_sum(cos: np.ndarray, N: int = iterations):
        output = np.zeros_like(cos)
        for n in range(N):
            output += out_summation_term(n, cos)
        return output

    def in_summation_term(k: int, cos):
        return (-1)**k*eval_legendre(2*k+1, cos)/(1-4*k**2)

    def in_sum(cos: np.ndarray, N: int = iterations):
        output = np.zeros_like(cos)
        output += np.pi/4+np.pi*eval_legendre(2, cos)/4
        for n in range(N):
            output -= in_summation_term(n, np.abs(cos))
        return output

    plt.plot(x_values, out_sum(cos_values), label="outer potential", alpha=0.5, color="b")
    plt.plot(x_values, in_sum(cos_values), label="inner potential", alpha=0.5, color="r")
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi[\sigma_0 a/2\epsilon_0]$")
    plt.savefig("potential.pdf", bbox_inches="tight")


def p_set_7_no_2c():
    resolution: int = 1000
    iterations: int = 1000
    cos_values = np.linspace(-1, 1, resolution)

    def summation_term(index: int, ad, cos):
        return -((index * (1 + 2*index)) / (3 * index + 1)) * ad**(index + 1) * eval_legendre(index, cos)

    def distribution_sum(cos: np.ndarray, ad: float, N: int = iterations):
        output = np.zeros_like(cos)
        for n in range(0, N):
            output += summation_term(n, ad, cos)
        return output/4

    def save_plot(name: str):
        plt.legend()
        plt.xlabel(r"$\cos\theta$")
        plt.ylabel(r"$\sigma[q/\pi a^2]$")
        plt.savefig(f"{name}.pdf", bbox_inches="tight")
        plt.clf()
        print("done")

    for ad in [1/5, 4/5]:
        plt.plot(cos_values, distribution_sum(cos_values, ad), label=f"$a/d={int(ad*5)}/5$")

    save_plot("charge_dist")

    smaller_plot = distribution_sum(cos_values, 1/5)
    larger_plot = distribution_sum(cos_values, 4/5)
    plt.ylim(1.1*min(smaller_plot), 1.1*max(larger_plot))
    for ad in [1/5, 4/5]:
        plt.plot(cos_values, distribution_sum(cos_values, ad), label=f"$a/d={int(ad*5)}/5$")
    save_plot("charge_dist_zoom")


if __name__ == '__main__':
    pass
