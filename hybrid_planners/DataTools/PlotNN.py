import numpy as np
import matplotlib.pyplot as plt

def make_mod_plots(vehicle):
    path  = vehicle + "ArchHistory/ModHistory_Lap_"

    for i in range(10):
        lap_path  = path + str(i) + ".npy"

        arr = np.load(lap_path)
        pp, nn, steering = arr[:, 0], arr[:, 1], arr[:, 2]
        steering = np.clip(steering, -0.45, 0.45)

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(pp)
        axs[0].set_title("Pure Pursuit")
        axs[0].set_ylim(-0.5, 0.5)
        axs[0].grid()

        axs[1].plot(nn * 0.4)
        axs[1].set_ylim(-0.5, 0.5)
        axs[1].set_title("Neural Network")
        axs[1].grid()

        axs[2].plot(steering)
        axs[2].set_title("Steering Angle")
        axs[2].set_ylim(-0.5, 0.5)
        axs[2].grid()

        plt.tight_layout()

        plt.savefig(vehicle + "ArchHistory/ModHistory_LapPlot_" + str(i) + ".svg", bbox_inches='tight', pad_inches=0)

        plt.show()
        

def make_mod_plots_thesis(vehicle):
    path  = vehicle + "ArchHistory/ModHistory_Lap_"

    y_lim = 0.48
    for j in range(10):
        lap_path  = path + str(j) + ".npy"

        arr = np.load(lap_path)
        pp, nn, steering = arr[:, 0], arr[:, 1], arr[:, 2]
        steering = np.clip(steering, -0.45, 0.45)

        fig, axs = plt.subplots(2, 1, figsize=(6.5, 3))
        axs[0].plot(pp, label="PP", color="darkblue")

        axs[0].plot(nn * 0.4, label="NN", color='r', alpha=0.8)
        axs[0].set_ylim(-y_lim, y_lim)
        axs[0].set_ylabel("Steering")
        axs[0].grid()
        axs[0].legend(loc='upper right', ncol=2)

        b_label, g_label = False, False
        for i in range(len(steering)-1):
            if abs(nn[i]*0.4) > 0.15 and not g_label:
                axs[1].plot([i, i+1], steering[i:i+2], 'darkgreen', linewidth=1.5, label=">0.15")
                g_label = True
            elif not b_label:
                axs[1].plot([i, i+1], steering[i:i+2], 'b', linewidth=1.5, label="<0.15")
                b_label = True
            elif abs(nn[i]*0.4) > 0.15:
                axs[1].plot([i, i+1], steering[i:i+2], 'darkgreen', linewidth=1.5)
            else:
                axs[1].plot([i, i+1], steering[i:i+2], 'b', linewidth=1.5)
        axs[1].set_ylabel("Steering")
        axs[1].set_ylim(-y_lim, y_lim)
        axs[1].grid()
        axs[1].legend(loc='upper right', ncol=1)
        

        plt.tight_layout()

        plt.savefig(vehicle + "ArchHistory/ModHistory_LapPlot_" + str(j) + ".svg", bbox_inches='tight', pad_inches=0.01)
        plt.savefig(vehicle + "ArchHistory/ModHistory_LapPlot_" + str(j) + ".pdf", bbox_inches='tight', pad_inches=0.01)

        # plt.show()
        

def main():
    vehicle = "Data/Vehicles/FastTests2/Mod_columbia_small_2_1/"

    # make_mod_plots(vehicle)
    make_mod_plots_thesis(vehicle)

main()
