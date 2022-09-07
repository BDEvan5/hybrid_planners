import numpy as np
import matplotlib.pyplot as plt
import glob


def generate_summary_table():
    folder = "Data/Vehicles/SlowTests/"
    # folder = "Data/Vehicles/FastTests/"

    # map_name = "columbia_small"
    map_name = "f1_aut"
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
    folder = "Data/Vehicles/SlowTests/"
    # folder = "Data/Vehicles/FastTests/"

    # map_name = "columbia_small"
    map_name = "f1_aut"
    lap_n = 3
    folders = glob.glob(folder + f"*_{map_name}_1_{lap_n}/")
    agents = [f.split("/")[-2] for f in folders]
    agents.sort()
    print(agents)

    metrics = ["Total Distnce m", "Total Curvature m$^{-1}$", "Total Deviation m", "Avg. Progress \%", "Completion Rate \%"]
    inds = [2, 4, 6, 7, 10]
    agent_names = ["E2e", "Mod", "Serial"]
    
    with open(folder + f"ThesisTable_{map_name}.txt", 'w') as summary_file:
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
    
    with open(folder + f"ThesisTable_{map_name}.txt", 'a') as summary_file:
        for j in range(len(metrics)):
            summary_file.write(f"{metrics[j]} & ".ljust(30)  + " & ".join([f"{float(dt[i][inds[j]])} ".ljust(10) for i in range(len(agent_names))]) + "\\\\ \n")
        summary_file.write(f"\hline\n")
    


# generate_summary_table()

generate_condensed_table()