import matplotlib.pyplot as plt

def plot_energy_mix(t, x, title):
    labels = ["Fossil", "Nuclear", "Wind", "Solar", "Hydro"]

    for i in range(5):
        plt.plot(t, x[:, i], label=labels[i])

    plt.xlabel("Time")
    plt.ylabel("Share of electricity generation")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
