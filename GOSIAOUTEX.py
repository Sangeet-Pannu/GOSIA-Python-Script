import pandas as pd
import os
import math 
from pathlib import Path  
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import threading
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors  # Importing mplcursors for hover functionality 
import colorsys

import warnings
warnings.filterwarnings("ignore")

#----------------------------------|
# Created by Sangeet-Pal Pannu
# Affiliation: University of Guelph
# Date: Nov 21 2024
#----------------------------------|

#THINGS TO NOTE WHEN RUNNING THIS CODE:

# 1. the GLS level scheme must be pre-made and must match the GOSIA input level energy values.
#    a. Remember to UPDATE the GLS file everytime you add a new level or transition to GOSIA.

# 2. file_path must be changed if the GOSIA output file is named different

# 3. use run_GOSIA2RAD.sh to run program for conversion of GOSIA entries to RADWARE level scheme.

# 4. Make sure all the above libraries are installed if not do:  pip install [LIBRARY NAME]

#NOTE THIS VALUE HAS TO BE UPDATED WHEN LOOKING AT ANOTHER NUCLEUS
#---------------------------|
A_Nucleus = 110 #Cd 110 Nucleus
#---------------------------|

Save_status = False

reDisplay = False

#---------------------------|

Minimization_Run_Num = 40

#---------------------------|

#---------------------------|

GSBand = [657,1542,2479,3275]
GamBand = [1475,2220,3064]
IntrBand = [1783,2250,2876,3791]
K3Band = [2355]



#---------------------------|
dt = datetime.datetime.today()

def ME_2_Wu(ME_Val,L2,SpinU,SpinD):
    ConV_WU =(10.0**(-2.0*L2))*(((1.2)**(2.0*L2))/(4.0*math.pi))*((3.0/(L2+3.0))**2.0)*(A_Nucleus**((2.0*L2)/3.0)) # Conversion of e2bL to Wu
    B_E2 = ((ME_Val)**2)/(2*SpinU+1)
    return B_E2/ConV_WU

def ME_1_Wu(ME_Val,L2,SpinU,SpinD):
    ConV_WU =(10**(-2*(L2-1)))*((10/math.pi)*((1.2)**((2*L2)-2)))*((3/(L2+3))**2)*(L2**(((2*A_Nucleus)-2)/3))# Conversion of e2bL to Wu
    B_E2 = ((ME_Val)**2)/(2*SpinU+1)
    return B_E2/ConV_WU

def ME_Q_0(ME_Val,L2,K):
    if(K==0):
        if L2 == 2:
            CleGoM = -0.534522484
        if L2 == 4:
            CleGoM = -0.509
        if L2 == 6:
            CleGoM = -0.504
        if L2 == 8:
            CleGoM = -0.502

    if(K==2):
        if L2 == 2:
            CleGoM = 0.534522484
        if L2 == 4:
            CleGoM = -0.203
        if L2 == 6:
            CleGoM = -0.360

    Qs = math.sqrt(16*math.pi/5)*(CleGoM/(math.sqrt((2*L2+1))))*ME_Val
    Q0 = Qs*(((L2+1)*(2*L2+3))/(3*(K**2)-L2*(L2+1)))
    return Q0

def extract_section(file_lines, start_marker, end_marker):
    pos_Start=[None]
    pos_End=[None]
    for i, line in enumerate(file_lines):
        if start_marker in line:
            pos_Start.append(i)
        if end_marker in line:
            pos_End.append(i)
    if pos_Start[-1] is not None:
        return file_lines[pos_Start[-1] + 1 : pos_End[-1]] # records the section of lines we are interested in.
    return []

def extract_section_P3(file_lines, start_marker, end_marker): # Definition comes from wanting the Level Energy definitions
    pos_Start=[None]
    pos_End=[None]
    for i, line in enumerate(file_lines):
        if start_marker in line:
            pos_Start.append(i)
        if end_marker in line:
            pos_End.append(i)
    if pos_Start[-1] is not None:
        return file_lines[pos_Start[-1] + 1 : pos_End[1]] # records the section of lines we are interested in.
    return []


def extract_section_P2(file_lines, start_marker, end_marker): # Definition comes from wanting GLS information
    pos_Start=[None]
    pos_End=[None]
    for i, line in enumerate(file_lines):
        if start_marker in line:
            pos_Start.append(i)
        if end_marker in line:
            pos_End.append(i)
    if pos_Start[-1] is not None:
        return file_lines[pos_Start[-1] : pos_End[-1]] # records the section of lines we are interested in.
    return []

def cyan_shades(num_shades=10):
    """
    Generate 'num_shades' distinct shades of cyan (RGB tuples),
    with the last entry being a shade of green.
    Hue ~0.5 is cyan, ~0.33 is green in HSV.
    """
    hue_cyan = 0.5     # Cyan
    hue_green = 0.33   # Green
    saturation = 1     # Full saturation
    color_list = []
    
    for i in range(num_shades):
        param = i / max(num_shades - 1, 1)
        value = 0.3 + 0.7 * param  # Brightness from dark to bright

        if i == num_shades - 1:
            # Last color: green
            r, g, b = colorsys.hsv_to_rgb(hue_green, saturation, 1)
        else:
            r, g, b = colorsys.hsv_to_rgb(hue_cyan, saturation, value)

        color_list.append((r, g, b))

    return color_list

def modify_gamma(lines, gamma_energy, new_energy, new_intensity):
    # Find gamma lines and modify specified gamma
    num = 0
    modified = False
    for i, line in enumerate(lines):
        if(num<=9):
            num = num + 1
            if line.startswith('     '):  # Identifies a gamma line based on the format of spaces given before the gamma.
                parts = line.split()
                for m in range(len(gamma_energy)):
                    if float(parts[1]) == round(gamma_energy[m]*1000,2):
                        if new_energy is not None:
                            parts[1] = f"{new_energy[m]:.3f}"  # Update energy
                        if new_intensity is not None:
                            parts[7] = f"{new_intensity[m]:.3f}"  # Update intensity
                        lines[i] = ' '
                        for part in parts:
                            lines[i] += '\t ' + part
                        lines[i] += '\n'                 
                        modified = True
                        break
        if(num>9):
            if line.startswith('    '):  # Identifies a gamma line based on the format of spaces given before the gamma.
                parts = line.split()
                #print(parts)
                num = num + 1
                for m in range(len(gamma_energy)):
                    if float(parts[1]) == round(gamma_energy[m]*1000,2):
                        if new_energy is not None:
                            parts[1] = f"{new_energy[m]:.3f}"  # Update energy
                        if new_intensity is not None:
                            parts[7] = f"{new_intensity[m]:.3f}"  # Update intensity
                        lines[i] = ' '
                        for part in parts:
                            lines[i] += '\t ' + part
                        lines[i] += '\n'                 
                        modified = True
                        break

    if not modified:
        raise ValueError(f"a gamma transition energy was not found. Check to See if the gls file energies match the GOSIA input file energies")

    return lines

def modify_Label(lines,number): #Modifies number of Text boxes in gls file, which depends on the Q values I have
    for i, line in enumerate(lines):
        if line.startswith('  '):  # Identifies a gamma line based on the format of spaces given before the gamma.
            parts = line.split()
            parts[4] = number
            lines[i] = '  '
            for part in parts:
                lines[i] += '\t ' + part
            
            lines[i] += '\n'                 
            break
    return lines

def add_text_gls(Q_vals): #Modifies number of Text boxes in gls file, which depends on the Q values I have
    fin_lines = []
    Label_Str1 = '** Label'+'\t\t\t\t'+'text  NChars'  
    Label_Str2 = '++     SizeX     SizeY PositionX PositionY'
    fin_lines.append(Label_Str1)
    fin_lines.append(Label_Str2)

    for i in range(len(Q_vals)):
        line = "    {}  {}eb\t\t\t\t\t\t{}&".format(i,Q_vals[i],len(Q_vals[i]))
        fin_lines.append(line)
        line = "++\t{}\t{}  {}\t{}".format(50.0,56.67,x_val,y_val)# SizeX SizeY PositionX PositionY       
        fin_lines.append(line)

    return fin_lines


def save(MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data,run_num = Minimization_Run_Num):
    counter = 0
    filename = './CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'

    while os.path.isfile(filename.format(run_num,counter)):
        counter = counter + 1
    filepath = Path('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter))  
    filepath.parent.mkdir(parents=True, exist_ok=True) #Checks to see if the folder already exists

    MATRIX_Data.to_csv('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t',na_rep=' ',index=False,header=False)
    Lifetime_Data.to_csv('./CSV_OUTPUT/Lifetime_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t',na_rep=' ',index=False,header=False)
    MixingRatio_Data.to_csv('./CSV_OUTPUT/MixingRatio_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t',na_rep=' ',index=False,header=False)
    Yields_Data.to_csv('./CSV_OUTPUT/Yields_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t',na_rep=' ',index=False,header=False)
    BR_Data.to_csv('./CSV_OUTPUT/BR_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t',na_rep=' ',index=False,header=False)

    return counter

def display_prev(value,run_num = Minimization_Run_Num):
    """Function to display the entered number when the Display button is clicked."""
    try:
        counter = value
        filename = './CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter)

        
        MATRIX_Data=pd.read_csv('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        Lifetime_Data=pd.read_csv('./CSV_OUTPUT/Lifetime_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        MixingRatio_Data=pd.read_csv('./CSV_OUTPUT/MixingRatio_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        Yields_Data=pd.read_csv('./CSV_OUTPUT/Yields_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        BR_Data=pd.read_csv('./CSV_OUTPUT/BR_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        return MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data
    except ValueError:
        messagebox.showerror("Error", "Where are the saved files? or Are you sure {} is a good number?".format(value))


def Lifetime_Plot(ax,cleaned_data,run_label,color_calc):
    Index = []
    LT_Cal = []

    for index, row in cleaned_data.iterrows():
        Index.append(int(row["LEVEL"]))
        LT_Cal.append(float(row["LIFETIME(PSEC)"]))

    scatter_plot = ax.scatter(
        Index, 
        LT_Cal, 
        label=f"Calculated ({run_label})", 
        color=color_calc, 
        marker="o", 
        s=30
    )
    
    return scatter_plot

def BR_PLOT(ax,cleaned_data,run_label,color_calc):
    Index = []
    LT_Cal = []

    for index, row in cleaned_data.iterrows():
        Index.append(str(int(row["NS1"]))+'-->'+str(int(row["NF1"])))
        LT_Cal.append(float(row["(EXP-CAL)/ERROR"]))

    scatter_plot = ax.scatter(
        Index, 
        LT_Cal, 
        label=f"Calculated ({run_label})", 
        color=color_calc, 
        marker="o", 
        s=30
    )
    
    return scatter_plot

def MixingRatio_Plot(ax,cleaned_data,run_label,color_calc):
    MIX_Label= []
    MIX_Cal = []
    num = 0

    for index, row in cleaned_data.iterrows():
        MIX_Label.append(str(row["TRANSITION"])+str(row["EXP.DELTA"]))
        MIX_Cal.append(float(row["SIGMA"]))

    scatter_plot = ax.scatter(
        MIX_Label, 
        MIX_Cal, 
        label=f"Calculated ({run_label})", 
        color=color_calc, 
        marker="o", 
        s=30
    )

    return scatter_plot

def Yield_Plot(ax,cleaned_data,run_label,color_calc):

    GY_Label= []
    GY_Cal = []
    highlight_indices = []
    num = 0


    for index, row in cleaned_data.iterrows():
        try:
            # Try to read ENERGY(MEV) and DIFF. normally
            try:
                energy = float(row["ENERGY(MEV)"]) * 1000
            except (ValueError, TypeError):
                raw_energy = str(row.iloc[8]).strip()
                if "+" in raw_energy:
                    raw_energy = raw_energy.split("+")[0]
                energy = float(raw_energy) * 1000

            try:
                diff = float(row["DIFF."])
            except (ValueError, TypeError):
                diff = float(row.iloc[12])

            GY_Label.append((int(energy)))
            GY_Cal.append(diff)

        except Exception as e:
            #print(f"Skipped row due to error: {e}")
            continue

    paired_lists = list(zip(GY_Label, GY_Cal))

    # Sort the pairs by the first element (from list1)
    sorted_pairs = sorted(paired_lists, key=lambda pair: pair[0])

    # Unzip the sorted pairs back into two lists
    sorted_list1, sorted_list2 = zip(*sorted_pairs)

    # Convert the tuples back to lists (if needed)
    sorted_list1 = list(sorted_list1)
    sorted_list2 = list(sorted_list2)
    sorted_list1 = list(map(str, sorted_list1))
    sorted_list2 = [round(num, 2) for num in sorted_list2]

    scatter_plot = ax.bar(
        sorted_list1, 
        sorted_list2,
        width=0.6,
        alpha=0.7,

        color=color_calc,
        label=f"Calculated ({run_label})",  
    )

    return scatter_plot

def ME_Plot(ax,cleaned_data,run_label,color_calc):

    ME_Label= []
    ME_Cal = []
    highlight_indices = []
    num = 0
    for index, row in cleaned_data.iterrows():
        ME_Label.append(str(row["NI"]) +"-->"+ str(row["NF"]))
        ME_Cal.append(float(row["ME"]))

    scatter_plot = ax.scatter(
        ME_Label, 
        ME_Cal, 
        label=f"Calculated ({run_label})", 
        color=color_calc, 
        marker="o", 
        s=30
    )

    return scatter_plot

#-----------------------------------------------------------------------------------------------|
#--------Function Definitions Above --------||

# Load the file
file_path = '110Cd.out'
i=0
with open(file_path, 'r') as file:
    file_content = file.readlines()

RED = "\033[91m"
RESET = "\033[0m"

print(RED + "\n" + "#" * 60)
print(f"                 MINIMIZATION RUN NUMBER: {Minimization_Run_Num}")
print("#" * 60 + "\n" + RESET)


#----------------------------[Secton: Grab sections of interest from GOSIA output]--------------------------||
recorded_sections = {
    "Yields":{"start": "CALCULATED AND EXPERIMENTAL YIELDS", "end": "EXP. AND CALCULATED BRANCHING RATIOS"},
    "BR":{"start": "EXP. AND CALCULATED BRANCHING RATIOS", "end": "E2/M1 MIXING RATIOS"},
    "E2/M1 MIXING RATIOS": {"start": "E2/M1 MIXING RATIOS", "end": "CALCULATED LIFETIMES"},
    "Lifetimes":{"start": "CALCULATED LIFETIMES", "end": "CALCULATED AND EXPERIMENTAL MATRIX ELEMENTS"},
    "ME_Cal":{"start": "CALCULATED AND EXPERIMENTAL MATRIX ELEMENTS", "end": "MATRIX ELEMENTS"},
    "MATRIX ELEMENTS":{"start": "MATRIX ELEMENTS", "end": "********* END OF EXECUTION **********"}
}

Levels = {
    "Levels":{"start": "LEVELS", "end": "MATRIX ELEMENTS"}
    }

parsed_data_with_skipping = {}
for section, markers in recorded_sections.items(): #.items() pulls the entire dictionary together.
    extracted_lines = extract_section(file_content, markers["start"], markers["end"])
    parsed_data_with_skipping[section] = extracted_lines # creates another library with each sections information.

level_data = {}
for section, markers in Levels.items(): #.items() pulls the entire dictionary together.
    extracted_lines = extract_section_P3(file_content, markers["start"], markers["end"])
    level_data[section] = extracted_lines # creates another library with each sections information.

Level_index = [] # Records the level Energy where the index of the list is the index placed in GOSIA
Spin_index = []
for word in level_data["Levels"]:
    if len(word)>1:
        Level_index.append(word.split()[3]);
        Spin_index.append(word.split()[2]);
#----------------------------[Secton: Grab sections of interest from GOSIA output]--------------------------||

#--------[Secton: Clean data of whitespaces]--------------------------||
cleaned_data = {}
CleanedList=[]
CopyedList=[]
for section, lines in parsed_data_with_skipping.items():
    for i in lines:
        j = i.strip()
        if(len(j)>0):
            string = j.split()
            CleanedList.append(string)
    CopyedList = CleanedList.copy()
    cleaned_data[section]=CopyedList;
    CleanedList.clear()
#--------[Secton: Clean data of whitespaces]--------------------------||


#--------[ Section Data for Each recorded section]--------------------------||

Yields_Data=pd.DataFrame.from_dict(cleaned_data["Yields"])

BR_Data=pd.DataFrame.from_dict(cleaned_data["BR"])

MixingRatio_Data=pd.DataFrame.from_dict(cleaned_data["E2/M1 MIXING RATIOS"])

Lifetime_Data=pd.DataFrame.from_dict(cleaned_data["Lifetimes"])

MATRIX_Data=pd.DataFrame.from_dict(cleaned_data["ME_Cal"])

Cal_counter=save(MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data)
#--------[ Section Data for Each recorded section]--------------------------||


#--------[ Section: LOGIC for matrix element to transition determiniation]--------------------------||

Energy = []
M_Ele_E2 = []
M_Ele_M1 = []
Norm_BE2 = []
Q_vals = []
Status = False
for Col in cleaned_data["MATRIX ELEMENTS"]:
    if Col[0] == 'MULTIPOLARITY=2':
        Status = True

    if Col[0] == 'MULTIPOLARITY=7':
        Status = False

    if(Status): 
        if len(Col) > 1 and len(Col) < 9:                 # makes sure we are not trying to grab lists that dont have more than 2 columns or have all string entries.
            Lv1 = float(Level_index[int(Col[2])])           # INITIAL LEVEL ENERGY FROM WHICH TRANSITION ORIGINATES
            Lv2 = float(Level_index[int(Col[1])])           # FINAL LEVEL ENERGY FROM WHICH TRANSITION GOES TO

            if(Lv2 > Lv1):
                Spin_up = float(Spin_index[int(Col[1])])        # INITIAL LEVEL SPIN
                Spin_down = float(Spin_index[int(Col[2])])      # FINAL LEVEL SPIN
            else:
                Spin_up = float(Spin_index[int(Col[2])])        # INITIAL LEVEL SPIN
                Spin_down = float(Spin_index[int(Col[1])])      # FINAL LEVEL SPIN
            
            Energy.append(abs(Lv1-Lv2))
            
                                # We have the Energy of the transition
            if(abs(Lv1-Lv2)==0):
                if(Lv1*1000 == 1475 or Lv1*1000 == 2220 or Lv1*1000 == 3064): #This is to catch the K=2 band Terms.
                    Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),2),Spin_up,float(Col[3])))    # Q_value
                else:
                    Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),0),Spin_up,float(Col[3])))
            M_Ele_E2.append(ME_2_Wu(float(Col[3]),2.0,Spin_up,Spin_down))
            M_Ele_M1.append(0)
    if(Status==False):
        if len(Col) > 1 and len(Col) < 9:                    # makes sure we are not trying to grab lists that dont have more than 2 columns or have all string entries.
            Lv1 = float(Level_index[int(Col[2])])            # INITIAL LEVEL ENERGY FROM WHICH TRANSITION ORIGINATES
            Lv2 = float(Level_index[int(Col[1])])            # FINAL LEVEL ENERGY FROM WHICH TRANSITION GOES TO            
            if(Lv2 > Lv1):
                Spin_up = float(Spin_index[int(Col[1])])        # INITIAL LEVEL SPIN
                Spin_down = float(Spin_index[int(Col[2])])      # FINAL LEVEL SPIN
            else:
                Spin_up = float(Spin_index[int(Col[2])])        # INITIAL LEVEL SPIN
                Spin_down = float(Spin_index[int(Col[1])])      # FINAL LEVEL SPIN

            
            #indexM = Energy.index(round(abs(Lv1-Lv2),3))
            #M_Ele_M1[indexM]=ME_1_Wu(float(Col[3]),1.0,Spin_up,Spin_down)



# Prints the Transition Matrix Element in B(E2) Down [W.u]

CYAN  = "\033[96m"
GREEN = "\033[92m"
BOLD  = "\033[1m"


# ─────────────────────────────────────────────────────────
# 1) GOSIA OUTPUT INFORMATION
# ─────────────────────────────────────────────────────────
print(
    RED + "#" * 60 + "\n" + 
    BOLD + "                 GOSIA OUTPUT INFORMATION" + RESET + RED + "\n" +
    "#" * 60 + RESET + "\n"
)

# ─────────────────────────────────────────────────────────
# 2) B(M1) W.u
# ─────────────────────────────────────────────────────────
print(
    CYAN + "|-------------- B(M1) W.u --------------|\n" + RESET
)

# Header for the B(M1) table
print(f"{'Energy (keV)':>12s}   {'B(M1) (W.u)':>12s}")
print("-" * 30)

# Print only if M1 > 0
for i in range(len(M_Ele_E2)):
    if M_Ele_M1[i] > 0:
        # Example:  800.00        0.3000
        print(f"{Energy[i] * 1000:12.2f}   {M_Ele_M1[i]:12.4f}")

print("\n\n")

# ─────────────────────────────────────────────────────────
# 3) Intrinsic Quadrupole Moments           THE ORDERING HAS BEEN HARD CODED TO PRESENT VALUES IN BAND ORDER
# ─────────────────────────────────────────────────────────
print(
    CYAN + "|---------- Intrinsic Quadrupole Moments ----------|\n" + RESET
)

# Print a table header for Q-values
print(f"{'Energy (keV)':>12s} | {'Spin':>5s} | {'ME':>5s} | {'Q0':>5s}")
print("-" * 37)

print("Ground State Band")
for level_energy, q0, spin, me in Q_vals:
    # Right-align numeric fields in a fixed width for neat columns
    if(level_energy*1000 in GSBand):
        print(
            f"{(level_energy * 1000):12.2f} | "
            f"{spin:>5.2f} | "
            f"{me:>5.2f} | "
            f"{q0:>5.2f}"
        )
print("Gamma Band")
for level_energy, q0, spin, me in Q_vals:
    if(level_energy*1000 in GamBand):
        print(
            f"{(level_energy * 1000):12.2f} | "
            f"{spin:>5.2f} | "
            f"{me:>5.2f} | "
            f"{q0:>5.2f}"
        )
print("Intruder Band")
for level_energy, q0, spin, me in Q_vals:
    if(level_energy*1000 in IntrBand):
        print(
            f"{(level_energy * 1000):12.2f} | "
            f"{spin:>5.2f} | "
            f"{me:>5.2f} | "
            f"{q0:>5.2f}"
        )
print("K = 0_3+")
for level_energy, q0, spin, me in Q_vals:
    if(level_energy*1000 in K3Band):
        print(
            f"{(level_energy * 1000):12.2f} | "
            f"{spin:>5.2f} | "
            f"{me:>5.2f} | "
            f"{q0:>5.2f}")
print("Other Levels")
for level_energy, q0, spin, me in Q_vals:
    if(level_energy*1000 not in GSBand and level_energy*1000 not in GamBand and level_energy*1000 not in IntrBand and level_energy*1000 not in K3Band):
        print(
            f"{(level_energy * 1000):12.2f} | "
            f"{spin:>5.2f} | "
            f"{me:>5.2f} | "
            f"{q0:>5.2f}")
        

print("\n\n")

#--------[ Section: LOGIC for matrix element to transition determiniation]--------------------------||


#--------[ Section: LOGIC for writing B(E2) values to GLS file (ASCII)]--------------------------||

# User MUST HAVE a working gls file with all levels and transitions added as in GOSIA

#Reads the ags file which is produced from gls_conv (OPTION 3)
with open('/home/sangeetpannu/Phd_110Cd_Project/110_Cd_GOSIA.ags', 'r') as file:
        lines = file.readlines()

recorded_sections = {
    "StartOfile" :{"start": "** ASCII", "end": "** Gamma"},
    "Gamma":{"start": "** Gamma ", "end": "** Label"}
}

ascii_gls_file = {}
for section, markers in recorded_sections.items(): #.items() pulls the entire dictionary together.
    extracted_lines = extract_section_P2(lines, markers["start"], markers["end"])
    ascii_gls_file[section] = extracted_lines # creates another library with each sections information.


#This line alters the gls files gamma energy value and intensity
first_lines=modify_Label(lines=ascii_gls_file["StartOfile"],number='2')
test_lines=modify_gamma(lines=ascii_gls_file["Gamma"],gamma_energy=Energy, new_energy=M_Ele_E2, new_intensity=M_Ele_E2)

file = open('items.ags','w')
for item in first_lines:
    file.write(item)
for item in test_lines:
    file.write(item)
file.close()

#--------[ Section: LOGIC for writing B(E2) values to GLS file (ASCII)]--------------------------||

prev_Data = []

# MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data return values from display_prev.

for t in range(Cal_counter+1):
    prev_Data.append(display_prev(t,run_num = Minimization_Run_Num))
#--------[ Section: Plotting Diagrams/Graphs ]--------------------------||

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

fig_ME, ax_ME = plt.subplots(figsize=(10, 6))

ax_LT = axes[0, 0]  # Lifetimes
ax_BR = axes[0, 1]  # Branching Ratios
ax_GY = axes[1, 0]  # Yields
ax_MX = axes[1, 1]  # Mixing Ratios

shades = cyan_shades(Cal_counter+1)  # e.g., 5 distinct cyan shades
#----------[Lifetime Graphs]--------|
Index = []
LT_Exp = []
LT_Err = []

for Col in cleaned_data["Lifetimes"]:
    if len(Col) <= 2:
        Index.append(int(Col[0]))
        LT_Exp.append(np.nan)
        LT_Err.append(0)
    if len(Col) > 4:
        Index.append(int(Col[0]))
        LT_Exp.append(float(Col[2]))
        LT_Err.append(abs(float(Col[3])))

errorbar_plot = ax_LT.errorbar(
        Index, 
        LT_Exp, 
        yerr=LT_Err, 
        label=f"Experimental", 
        color="red", 
        fmt='o',
        ecolor='gray',
        capsize=3,
        elinewidth=2

        )
scatter_plot_Ex = ax_LT.scatter(
        Index, 
        LT_Exp, 
        color="red", 
        marker='o'
        )
cursor = mplcursors.cursor([scatter_plot_Ex], hover=mplcursors.HoverMode.Transient,multiple=False,highlight=True)

# METHOD TO ALLOW HOVERING ANNOTATION.... 

scatter_plot = []

#Calculated Run
for x in range(Cal_counter+1):
    scatter_plot.append(Lifetime_Plot(ax_LT,
    prev_Data[x][1],
    run_label=f"Mini.# :{x}",
    color_calc=shades[x],
    ))
    cursor = mplcursors.cursor([scatter_plot[x]], hover=mplcursors.HoverMode.Transient,multiple=False,highlight=True)

@cursor.connect("add")
def on_add(sel):
    # Identify whether the point is experimental or calculated
    if sel.artist == scatter_plot_Ex:
        sel.annotation.set_text(
            f"Exp:\nLEVEL={sel.target[0]:.1f} keV\nLifetime={sel.target[1]:.4f} ps"
        )
    elif sel.artist == scatter_plot:
        sel.annotation.set_text(
            f"Calc:\nLEVEL={sel.target[0]:.1f} keV\nLifetime={sel.target[1]:.4f} ps"
        )
    sel.annotation.arrow_patch.set_edgecolor("white")
    sel.annotation.arrow_patch.set_facecolor("white")

ax_LT.set_xlabel('Level Energy (keV)')
ax_LT.set_ylabel('Lifetimes (Pico-Seconds)')
ax_LT.set_yscale("log")
ax_LT.set_title('GOSIA: Lifetimes (CALCULATED VS. EXPERIMENTAL)')
ax_LT.legend()

#---[Branching ratio Graphs]--------|
BR_Label= []
BR_Exp = []
BR_Err = []
BR_Cal = []
num = 0

for Col in cleaned_data["BR"]:
    num = num + 1
    if num > 1:
        BR_Label.append(Col[0]+'-->'+Col[1])
        BR_Cal.append(float(Col[7]))
        BR_Exp.append(float(Col[4]))
        BR_Err.append(abs(float(Col[5])))


for x in range(Cal_counter+1):
    BR_PLOT(ax_BR,
    prev_Data[x][4],
    run_label=f"Mini.# :{x}",
    color_calc=shades[x],
    )

ax_BR.errorbar(BR_Label, BR_Exp,yerr = BR_Err,label= "Experimental", color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );


ax_BR.set_xticklabels(ax_BR.get_xticklabels(), rotation=40, ha="right")
ax_BR.set_ylabel('Branching Ratio')
# plot title
ax_BR.set_title('GOSIA: Branching Ratios (CALCULATED VS. EXPERIMENTAL)')
# showing legend
ax_BR.legend()


#------------[Yields Graphs]--------|
GY_Label= []
GY_Exp = []
GY_Err = []
GY_PErr = []
highlight_indices = []
num = 0

for row in cleaned_data["Yields"]:
        try:
            # Try to read ENERGY(MEV) and DIFF. normally
            try:
                energy = float(row[4]) * 1000
            except (ValueError, TypeError):
                raw_energy = str(row[8]).strip()
                if "+" in raw_energy:
                    raw_energy = raw_energy.split("+")[0]
                energy = float(raw_energy) * 1000

            # --- DIFF Handling ---
            diff_val = str(row[8]).strip()
            if diff_val == "****":
                y = 20
            else:
                try:
                    y = float(diff_val)
                except ValueError:
                    y = float(row[12])  # fallback if still malformed

            GY_Label.append((int(energy)))
            GY_PErr.append(y)

        except Exception as e:
            #print(f"Skipped row due to error: {e}")
            continue

# Zip the lists together
paired_lists = list(zip(GY_Label, GY_PErr))

# Sort the pairs by the first element (from list1)
sorted_pairs = sorted(paired_lists, key=lambda pair: pair[0])

# Unzip the sorted pairs back into two lists
sorted_list1, sorted_list2 = zip(*sorted_pairs)

# Convert the tuples back to lists (if needed)
sorted_list1 = list(sorted_list1)
sorted_list2 = list(sorted_list2)
sorted_list1 = list(map(str, sorted_list1))
sorted_list2 = [round(num, 2) for num in sorted_list2]

for x in range(Cal_counter):
    Yield_Plot(ax_GY,
    prev_Data[x][3],
    run_label=f"Mini.# :{x}",
    color_calc=shades[x],
    )

ax_GY.bar(sorted_list1, sorted_list2,width=0.6, edgecolor='red',facecolor='none',alpha=1.0,hatch='////',label= "*Recent Calc*");
ax_GY.set_xticklabels(ax_GY.get_xticklabels(), rotation=40, ha="right")
y_min = min(sorted_list2) - 2  # Adjust margin as needed
y_max = max(sorted_list2) + 2


ax_GY.set_ylim(y_min, y_max)
# x-axis label
ax_GY.set_xlabel('Energy (keV)')
#ax_GY.set_yscale("log")   
# frequency label
ax_GY.set_ylabel('sigma')
# plot title
#ax_GY.set_title('% Di',x=-0.2,y=-0.2)
# showing legend
#ax_GY.legend()


#------[Mixing Ratio Graphs]--------|
MIX_Label= []
MIX_Exp = []
MIX_Err = []
MIX_Cal = []
num = 0
for Col in cleaned_data["E2/M1 MIXING RATIOS"]:
    num = num + 1
    if num > 1:
        MIX_Label.append(str(Col[0])+str(Col[1]))
        MIX_Cal.append(float(Col[3]))
        MIX_Exp.append(float(Col[2]))
        if(abs(float(Col[4])) != 0):
            error_cal = abs(float(Col[3])-float(Col[2]))/abs(float(Col[4]))

        else:
            error_cal = 0
        MIX_Err.append(error_cal)

ax_MX.errorbar(MIX_Label, MIX_Exp,label= "Experimental",yerr = MIX_Err, color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    )

for x in range(Cal_counter+1):
    MixingRatio_Plot(ax_MX,
    prev_Data[x][2],
    run_label=f"Mini.# :{x}",
    color_calc=shades[x],
    )

# x-axis label
ax_MX.set_xlabel('Li-->Lf')
#plt.yscale("log")   
# frequency label
ax_MX.set_ylabel('(E2/M1) Mixing Ratio')
# plot title
ax_MX.set_title('GOSIA: E2/M1 Mixing Ratios (CALCULATED VS. EXPERIMENTAL)',y=-0.2)
# showing legend
ax_MX.legend()

#---------[Matrix Element Comp]
ME_Label= []
ME_Exp = []
ME_Err = []
ME_Cal = []
highlight_indices = []
num = 0
for Col in cleaned_data["ME_Cal"]:
    num = num + 1
    if num > 1:
        ME_Label.append(Col[0] +"-->"+ Col[1])
        ME_Cal.append(float(Col[3]))
        ME_Exp.append(float(Col[2]))
        if(abs(float(Col[4])) != 0):
            ME_Err.append(abs(float(Col[3])-float(Col[2]))/abs(float(Col[4])))
        else:
            ME_Err.append(0)
         
        if(abs(float(Col[4])) >= 3.0):
            highlight_indices.append(num-2)
itr = 0
for idx in highlight_indices:
    itr = itr + 1
    ax_ME.annotate(
        text=f"{ME_Label[idx]}",  # Label with the x-value
        xy=(ME_Label[idx], ME_Cal[idx]),  # Point at the data
        xytext=(ME_Label[idx],ME_Cal[idx]-0.5),  # Position of the label (adjust as needed)
        arrowprops=dict(arrowstyle="->", color="white"),
        ha='center'  # Center-align the text
    )

ax_ME.errorbar(ME_Label, ME_Exp,label= "Experimental",yerr = ME_Err, color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps 
    elinewidth=2  # Line width of error bars
    );


for x in range(Cal_counter+1):
    ME_Plot(ax_ME,
    prev_Data[x][0],
    run_label=f"Mini.# :{x}",
    color_calc=shades[x],
    )

# x-axis label
ax_ME.set_xlabel('Energy (keV)')   
# frequency label
#ax_ME.set_xticks(rotation=90)
ax_ME.set_ylabel('Matrix Elements (e2bl)')
ax_ME.set_xticklabels(ax_ME.get_xticklabels(), rotation=40, ha="right")
# plot title
ax_ME.set_title('GOSIA: Given Matrix Elements (CALCULATED VS. EXPERIMENTAL)')
# showing legend
ax_ME.legend()

fig.tight_layout()
fig_ME.tight_layout()

plt.show()

#--------[ Section: Plotting Diagrams/Graphs ]--------------------------||


