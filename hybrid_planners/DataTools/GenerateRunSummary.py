import numpy as np
import matplotlib.pyplot as plt
import glob


def generate_summary_table():
    # folder = "Data/Vehicles/SlowTests/"
    folder = "Data/Vehicles/devel2fast/"
    # folder = "Data/Vehicles/FastTests/"

    map_name = "columbia_small"
    # map_name = "f1_aut"
    folders = glob.glob(folder + f"*_{map_name}*/")
    agents = [f.split("/")[-2] for f in folders]
    agents.sort()
    print(agents)
    
    for i, agent in enumerate(agents):
        with open(folder + f"{agent}/SummaryStatistics.txt", 'r') as agent_file:
            lines = agent_file.readlines()
            if i == 0:
                with open(folder + f"RunSummaryTable_{map_name}.txt", 'w') as summary_file:
                    summary_file.write(f"Agent ".ljust(30)  + lines[1])
        
            with open(folder + f"RunSummaryTable_{map_name}.txt", 'a') as summary_file:
                summary_file.write(f"{agent} ".ljust(30)  + lines[2])


def generate_condensed_table():
    # folder = "Data/Vehicles/SlowTests/"
    folder = "Data/Vehicles/BigSlowTests/"
    # folder = "Data/Vehicles/FastTests/"

    # map_name = "columbia_small"
    map_name = "f1_gbr"
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
    

def generate_summaries():
    folder = "Data/Vehicles/BigSlowTests/"

    # map_name = "f1_gbr"
    map_name = "f1_aut"
    for run_n in range(10):
        folders = glob.glob(folder + f"*_{map_name}_1_{run_n}/")
        agents = [f.split("/")[-2] for f in folders]
        agents.sort()
        print(agents)
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

def make_plots():
    folder = "Data/Vehicles/BigSlowTests/"

    map_name = "f1_aut"
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
                # for j in range(len(inds)):
                e2e_data[i-2].append(float(data[1]))
                mod_data[i-2].append(float(data[2]))
                serial_data[i-2].append(float(data[3]))
    # plot distances
    for i in range(5):
        plt.figure(i, figsize=(3, 2.4))
        plt.xlabel(metrics[i])
        plt.boxplot(e2e_data[i], positions=[1], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.boxplot(serial_data[i], positions=[3], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.boxplot(mod_data[i], positions=[2], widths=0.6, vert=False, boxprops={'linewidth':2, 'color':'darkblue'}, whiskerprops={'linewidth':3, 'color':'darkblue'}, medianprops={'linewidth':3, 'color':'darkblue'}, capprops={'linewidth':3, 'color':'darkblue'})
        plt.yticks([1, 2, 3], agent_names)

        plt.tight_layout()
        plt.grid(True)

        plt.savefig("Data/LowSpeedEval/" + f"RepeatTrain_{metric_names[i]}.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig("Data/LowSpeedEval/" + f"RepeatTrain_{metric_names[i]}.svg", bbox_inches='tight', pad_inches=0)

    plt.show()

def make_mean_table():
    folder = "Data/Vehicles/BigSlowTests/"

    map_name = "f1_aut"

    metrics = ["Total Distance", "Total Curvature", "Total Deviation", "Avg. Progress", "Completion Rate"]
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
                # for j in range(len(inds)):
                e2e_data[i-2].append(float(data[1]))
                mod_data[i-2].append(float(data[2]))
                serial_data[i-2].append(float(data[3]))


    big_table = folder + "thesis_mean_table.txt"
    with open(big_table, 'w') as summary_file:
        summary_file.write(" \\textbf{Metric} & \\textbf{".ljust(30) + "} &  \\textbf{".join([f"{agent} ".ljust(10) for agent in agent_names]) + "} \\\\ \n")
        summary_file.write("\\hline \n")
        summary_file.write("\\hline \n")
        for i in range(len(metrics)):

            summary_file.write(f"{metrics[i]},".ljust(20))
            summary_file.write(f" & {np.mean(e2e_data[i]):.2f}".ljust(10))
            summary_file.write(f" & {np.mean(serial_data[i]):.2f}".ljust(10))
            summary_file.write(f" & {np.mean(mod_data[i]):.2f} \\\\ \n".ljust(10))

        summary_file.write("\\hline \n")


generate_summary_table()

# generate_condensed_table()
# generate_summaries()
# convert_summaries_to_big_table()
# make_plots()
# make_mean_table()


