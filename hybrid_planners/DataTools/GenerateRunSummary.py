import numpy as np
import matplotlib.pyplot as plt
import glob


def generate_summary_table(folder):

    map_name = "columbia_small"
    # map_name = "f1_aut"
    folders = glob.glob(folder + f"*_{map_name}*/")
    print("Folders: ", folders)
    agents = [f.split("/")[-2] for f in folders]
    agents.sort()
    print(agents)
    
    for i, agent in enumerate(agents):
        try:
            with open(folder + f"{agent}/SummaryStatistics.txt", 'r') as agent_file:
                lines = agent_file.readlines()
                if i == 0:
                    with open(folder + f"RunSummaryTable_{map_name}.txt", 'w') as summary_file:
                        summary_file.write(f"Agent ".ljust(30)  + lines[1])
            
                with open(folder + f"RunSummaryTable_{map_name}.txt", 'a') as summary_file:
                    summary_file.write(f"{agent} ".ljust(30)  + lines[2])
        except: pass

def generate_condensed_table():
    # folder = "Data/Vehicles/SlowTests/"
    # folder = "Data/Vehicles/BigSlowTests/"
    folder = "Data/Vehicles/FastTests2/"

    map_name = "columbia_small"
    # map_name = "f1_gbr"
    # map_name = "f1_aut"
    run_n = 4
    folders = glob.glob(folder + f"*_{map_name}_1_{run_n}/")
    agents = [f.split("/")[-2] for f in folders]
    agents.sort()
    print(agents)

    metrics = ["Total Distnce m", "Total Curvature m$^{-1}$", "Total Deviation m", "Avg. Progress \%", "Completion Rate \%"]
    inds = [2, 4, 6, 7, 10]
    agent_names = ["E2e", "Mod", "Serial"]
    
    with open(folder + f"ThesisTable_{map_name}_{run_n}.txt", 'w') as summary_file:
        # summary_file.write(f"Agent ".ljust(30)  + lines[1])
        summary_file.write("\\textbf{ Metric } & \\textbf{".ljust(30) + "} & \\textbf{".join([f"{agent} ".ljust(10) for agent in agent_names]) + "}\\\\ \n")
        summary_file.write(f"\hline\n")
        summary_file.write(f"\hline\n")

    dt = []
    for i, agent in enumerate(agents):
        with open(folder + f"{agent}/SummaryStatistics.txt", 'r') as agent_file:
            lines = agent_file.readlines()
            data = lines[2].split(",")
            dt.append(data)
    
    with open(folder + f"ThesisTable_{map_name}_{run_n}.txt", 'a') as summary_file:
        for j in range(len(metrics)):
            summary_file.write(f"{metrics[j]} & ".ljust(30)  + " & ".join([f"{float(dt[i][inds[j]])} ".ljust(10) for i in range(len(agent_names))]) + "\\\\ \n")
        summary_file.write(f"\hline\n")
    

def generate_summaries(folder):
    map_name = "columbia_small"
    # map_name = "f1_aut"

    set_n = 1

    for run_n in range(10):
        folders = glob.glob(folder + f"*_{map_name}_{set_n}_{run_n}/")
        agents = [f.split("/")[-2] for f in folders]
        agents.sort()
        print(agents)
        # print(folders)
        file_name = folder + f"SummaryTable_{map_name}_{run_n}.txt"

        metrics = ["Total Distance", "Total Curvature", "Total Deviation", "Avg. Progress", "Completion Rate"]
        inds = [2, 4, 6, 7, 10]
        agent_names = ["E2e", "Mod", "Serial"]
        
        with open(file_name, 'w') as summary_file:
            # summary_file.write(f"Agent ".ljust(30)  + lines[1])
            summary_file.write(" Metric  ".ljust(30) + " ".join([f"{agent} ".ljust(10) for agent in agent_names]) + "} \n")
            summary_file.write(f"------------------------------\n")

        dt = []
        for i, agent in enumerate(agents):
            with open(folder + f"{agent}/SummaryStatistics.txt", 'r') as agent_file:
                lines = agent_file.readlines()
                data = lines[2].split(",")
                dt.append(data)
        
        with open(file_name, 'a') as summary_file:
            for j in range(len(metrics)):
                summary_file.write(f"{metrics[j]}  ,".ljust(30)  + ", ".join([f"{float(dt[i][inds[j]])} ".ljust(10) for i in range(len(agent_names))]) + " \n")

def convert_summaries_to_plot():
    folder = "Data/Vehicles/BigSlowTests/"

    map_name = "f1_aut"

    metrics = ["Total Distance", "Total Curvature", "Total Deviation", "Avg. Progress", "Completion Rate"]
    inds = [2, 3, 4, 5, 6]
    agent_names = ["E2e", "Mod", "Serial"]

    e2e_data = [[] for i in range(len(inds))]
    mod_data = [[] for i in range(len(inds))]
    serial_data = [[] for i in range(len(inds))] 

    for run_n in range(10):
        file_name = folder + f"SummaryTable_{map_name}_{run_n}.txt"
        
        with open(file_name, 'r') as summary_file:
            lines = summary_file.readlines()

            for i, line in enumerate(lines): # cycles through metrics
                if i == 0 or i == 1:   continue
                data = line.split(",")
                # for j in range(len(inds)):
                e2e_data[i-2].append(float(data[1]))
                mod_data[i-2].append(float(data[2]))
                serial_data[i-2].append(float(data[3]))

    big_table = folder + "big_table.txt"
    with open(big_table, 'w') as summary_file:
        summary_file.write(f"E2e \n")
        for i in range(len(metrics)):
            summary_file.write(f"{metrics[i]} ,".ljust(20))
            summary_file.write("".join([f"{float(e2e_data[i][j]):.2f} , ".ljust(10) for j in range(10)]) + "\n")

        
        summary_file.write(f" \n \nSerial \n")
        for i in range(len(metrics)):
            summary_file.write(f"{metrics[i]} ,".ljust(20))
            summary_file.write("".join([f"{float(serial_data[i][j]):.2f} , ".ljust(10) for j in range(10)]) + "\n")


        summary_file.write(f" \n \nMod \n")
        for i in range(len(metrics)):
            summary_file.write(f"{metrics[i]} ,".ljust(20))
            summary_file.write("".join([f"{float(mod_data[i][j]):.2f} , ".ljust(10) for j in range(10)]) + "\n")

def make_plots(folder):
    # folder = "Data/Vehicles/FastTests2/"

    # map_name = "f1_aut"
    map_name = "columbia_small"
    metrics = ["Total Distance m", "Total Curvature 1/m", "Total Deviation m", "Avg. Progress %", "Completion Rate %"]

    metric_names = ["TotalDistance", "TotalCurvature", "TotalDeviation", "AvgProgress", "CompletionRate"]
    inds = [2, 3, 4, 5, 6]
    agent_names = ["E2e", "Serial", "Mod"]

    e2e_data = [[] for i in range(len(inds))]
    mod_data = [[] for i in range(len(inds))]
    serial_data = [[] for i in range(len(inds))] 

    for run_n in range(10):
        file_name = folder + f"SummaryTable_{map_name}_{run_n}.txt"
        
        with open(file_name, 'r') as summary_file:
            lines = summary_file.readlines()

            for i, line in enumerate(lines): # cycles through metrics
                if i == 0 or i == 1:   continue
                data = line.split(",")
                e2e = float(data[1])
                mod = float(data[2])
                serial = float(data[3])
                if not np.isnan(e2e) and run_n!=8:
                    e2e_data[i-2].append(e2e)
                if not np.isnan(mod)  and run_n!=1:
                    mod_data[i-2].append(mod)
                if not np.isnan(serial)  and run_n!=6:
                    serial_data[i-2].append(serial)
    # plot distances
    for i in range(len(inds)):
        plt.figure(i, figsize=(3, 1.5))
        plt.xlabel(metrics[i])
        plt.boxplot(e2e_data[i], positions=[1], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.boxplot(serial_data[i], positions=[2], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.boxplot(mod_data[i], positions=[3], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.yticks([1, 2, 3], agent_names)

        plt.plot(np.mean(e2e_data[i]), 1, 'o', color='red', markersize=6)
        plt.plot(np.mean(serial_data[i]), 2, 'o', color='red', markersize=6)
        plt.plot(np.mean(mod_data[i]), 3, 'o', color='red', markersize=6)

        plt.tight_layout()
        plt.grid(True)

        plt.savefig("Data/ThesisEval/" + f"RepeatTrain_{metric_names[i]}.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig("Data/EvalAnalysis/" + f"RepeatTrain_{metric_names[i]}.svg", bbox_inches='tight', pad_inches=0)

    plt.show()

def make_mean_table(folder):

    map_name = "columbia_small"
    # map_name = "f1_aut"

    metrics = ["Total Distance m", "Total Curvature m$^{-1}$", "Total Deviation m", "Avg. Progress \%", "Completion Rate \%"]
    inds = [2, 3, 4, 5, 6]
    agent_names = ["E2e", "Serial", "Mod"]

    e2e_data = [[] for i in range(len(inds))]
    mod_data = [[] for i in range(len(inds))]
    serial_data = [[] for i in range(len(inds))] 

    for run_n in range(10):
        file_name = folder + f"SummaryTable_{map_name}_{run_n}.txt"
        
        with open(file_name, 'r') as summary_file:
            lines = summary_file.readlines()

            for i, line in enumerate(lines): # cycles through metrics
                if i == 0 or i == 1:   continue
                data = line.split(",")
                e2e = float(data[1])
                mod = float(data[2])
                serial = float(data[3])
                if not np.isnan(e2e) and run_n!=8:
                    e2e_data[i-2].append(e2e)
                if not np.isnan(mod)  and run_n!=1:
                    mod_data[i-2].append(mod)
                if not np.isnan(serial)  and run_n!=6:
                    serial_data[i-2].append(serial)
                # e2e_data[i-2].append(float(data[1]))
                # mod_data[i-2].append(float(data[2]))
                # serial_data[i-2].append(float(data[3]))


    big_table = folder + "thesis_mean_table.txt"
    with open(big_table, 'w') as summary_file:
        summary_file.write(" \\textbf{Metric} & \\textbf{".ljust(30) + "} &  \\textbf{".join([f"{agent} ".ljust(10) for agent in agent_names]) + "} \\\\ \n")
        summary_file.write("\\hline \n")
        summary_file.write("\\hline \n")
        for i in range(len(metrics)):

            summary_file.write(f"{metrics[i]}".ljust(28))
            summary_file.write(f" & {np.mean(e2e_data[i]):.2f} $\pm$ {np.std(e2e_data[i]):.2f}".ljust(25))
            summary_file.write(f" & {np.mean(serial_data[i]):.2f} $\pm$ {np.std(serial_data[i]):.2f}".ljust(25))
            summary_file.write(f" & {np.mean(mod_data[i]):.2f} $\pm$ {np.std(mod_data[i]):.2f} ".ljust(25) + "\\\\\n")
            # summary_file.write(f" & {np.mean(serial_data[i]):.2f}".ljust(10))
            # summary_file.write(f" & {np.mean(mod_data[i]):.2f} \\\\ \n".ljust(10))

        summary_file.write("\\hline \n")

path = "Data/Vehicles/FFT2/"
# path = "Data/Vehicles/ModTests1/"

# generate_summary_table(path)

# generate_condensed_table()
# generate_summaries(path)
# convert_summaries_to_big_table()
make_plots(path)
# make_mean_table(path)


