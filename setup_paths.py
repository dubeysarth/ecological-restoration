import sys
import os

def add_path():
    sys.path.insert(1,os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions'))

def get_paths():
    paths = []
    # 0
    paths.append(os.path.join(os.getcwd(), 'Code'))
    # 1
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions'))
    # 2
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution'))
    # 3
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '02_Generate_Database'))
    # 4
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '03_Scatter_Data'))
    # 5
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '04_Importance_Data'))
    # 6
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '05_Analysis'))
    # 7
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '06_Figures'))
    # 8
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '07_Miscellaneous'))
    # 9
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', '01_Real_Networks_1D'))
    # 10
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', '02_Synthetic_Networks_1D'))
    # 11
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', '03_Synthetic_Networks_2D'))
    # 12
    paths.append(os.path.join(os.getcwd(), 'Code', '01_Species_Reintroductions', '01_Simulation_Execution', '04_Real_Networks_2D'))

    return paths