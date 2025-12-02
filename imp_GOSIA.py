import pandas as pd
import os
import math 
from pathlib import Path  
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors 
import colorsys
import io 
import csv 
import re 
import sys 
import subprocess

import warnings
warnings.filterwarnings("ignore")

A_Nucleus = 110 
Minimization_Run_Num = 1

try:
    Minimization_Run_Num = Minimization_Run_Num
except NameError:
    Minimization_Run_Num = 1 

# Placeholders for GLS text box positioning, required by add_text_gls
x_val = 100.0
y_val = 100.0

# Data for K=0 (First 5 rows of Table 1, I_i = 2, 4, 6, 8, 10)
I_k0 = np.array([2, 4, 6, 8, 10])
q_k0_gamma_10 = np.array([-0.28, -0.35, -0.38, -0.39, -0.40])
q_k0_gamma_20 = np.array([-0.25, -0.24, -0.21, -0.20, -0.19])
q_k0_gamma_27_5 = np.array([-0.10, -0.05, -0.05, -0.05, -0.05])



# Data for K=2 (Remaining 9 rows of Table 1, I_i = 2, 3, 4, 5, 6, 7, 8, 9, 10)
I_k2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
q_k2_gamma_10 = np.array([0.28, 0.00, -0.15, -0.23, -0.33, -0.32, -0.36, -0.36, -0.41])
q_k2_gamma_20 = np.array([0.25, 0.00, -0.23, -0.20, -0.41, -0.26, -0.45, -0.27, -0.42])
q_k2_gamma_27_5 = np.array([0.10, 0.00, -0.30, -0.08, -0.15, -0.08, -0.10, -0.07, -0.09])


gamma_colors = {10: '#DC143C', 20: '#4682B4', 27.5: '#191970'}
model_colors = {'K0': '#008080', 'K2': '#FFA500'}
experimental_color = '#FF4B4B'

def calculated_quadrupole_moment(I, K):
    """
    Calculates the normalized spectroscopic quadrupole moment Q/Q_0 based on 
    the rotor model formula: (3K^2 - I(I+1)) / ((I+1)(2I+3))
    """
    # Ensure I is treated as a float to prevent integer division issues
    I_f = I.astype(float) 
    
    # Calculate the value using the provided equation
    val = (3 * K**2 - I_f * (I_f + 1)) / ((I_f + 1) * (2 * I_f + 3))
    return val

q_calculated_K0 = calculated_quadrupole_moment(I_k0, 0)
q_calculated_K2 = calculated_quadrupole_moment(I_k2, 2)

def cyan_shades(n):
    """Generates n distinct shades of cyan."""
    shades = []
    for i in range(n):
        # H=180 (cyan), S=50-100, L=30-70
        h = 180 / 360.0  # Hue for Cyan
        s = 0.5 + (0.5 * i / max(1, n - 1))  # Saturation
        l = 0.5 - (0.2 * i / max(1, n - 1))  # Lightness
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        shades.append((r, g, b))
    return shades

def ME_2_Wu(ME_Val,L2,SpinU,SpinD):
    ConV_WU =(10.0**(-2.0*L2))*(((1.2)**(2.0*L2))/(4.0*math.pi))*((3.0/(L2+3.0))**2.0)*(A_Nucleus**((2.0*L2)/3.0)) # Conversion of e2bL to Wu
    B_E2 = ((ME_Val)**2)/(2*SpinU+1)
    return B_E2/ConV_WU

def ME_3_Wu(ME_Val,L2,SpinU,SpinD):
    ConV_WU =(10.0**(-2.0*L2))*(((1.2)**(2.0*L2))/(4.0*math.pi))*((3.0/(L2+3.0))**2.0)*(A_Nucleus**((2.0*L2)/3.0)) # Conversion of e2bL to Wu
    B_E2 = ((ME_Val)**2)/(2*SpinU+1)
    return B_E2/ConV_WU

def ME_1_Wu(ME_Val,L2,SpinU,SpinD):
    ConV_WU =(10**(-2*(L2-1)))*((10/math.pi)*((1.2)**((2*L2)-2)))*((3/(L2+3))**2)*(L2**(((2*A_Nucleus)-2)/3))# Conversion of e2bL to Wu
    B_E2 = ((ME_Val)**2)/(2*SpinU+1)
    return (B_E2/ConV_WU)*(10**3)

def ME_Q_0(ME_Val,L2,K):
    CleGoM = 0
    if L2 == 2:
        CleGoM = 0.534522484
    if L2 == 3:
        CleGoM = 0.645497224
    if L2 == 4:
        CleGoM = 0.713506068
    if L2 == 6:
        CleGoM = 0.792825
    if L2 == 8:
        CleGoM = 0.837708
    Qs = math.sqrt(16*math.pi/5)*(CleGoM/(math.sqrt((2*L2+1))))*ME_Val
    Q0 = Qs*(((L2+1)*(2*L2+3))/(3*(K**2)-L2*(L2+1)))
    return Q0

def ME_Q_ratios(Tra_ME,Dia_ME,L2):
    #print(f'| Transitional ME {Tra_ME} | Diagonal ME {Dia_ME} | Spin of ME {L2} | ')
    CleGoM = 0
    CleGoMQ0 = 0
    if L2 == 2:
        CleGoMQ0 = 0.534522484
        CleGoM = 0.447214
    if L2 == 3:
        CleGoM = 0.645497224
    if L2 == 4:
        CleGoMQ0 = 0.713506068
        CleGoM = 0.534522
    if L2 == 6:
        CleGoMQ0 = 0.792825
        CleGoM = 0.560968
    if L2 == 8:
        CleGoMQ0 = 0.837708
        CleGoM = 0.532554

    Qts = (CleGoM*CleGoMQ0)*(Dia_ME/Tra_ME)
    return Qts

def ME_Q_ratios2(Tra_ME,Dia_ME,L2):
    CleGoM = 0
    CleGoMQ0 = 0
    if L2 == 4:
        CleGoMQ0 = 0.713506068
        CleGoM = 0.345033
    if L2 == 6:
        CleGoMQ0 = 0.792825
        CleGoM = 0.484732
    if L2 == 8:
        CleGoMQ0 = 0.837708
        CleGoM = 0.573944

    Qts = (CleGoM*CleGoMQ0)*(Dia_ME/Tra_ME)
    return Qts

GSBand = [0,657,1542,2479,3275]
GamBand = [1475,2220,3064]
IntrBand = [1473,1783,2250,2876,3791]
K3Band = [1731,2355]
K4Band = [2561]
NegParBand = [2078]


# ----------------------------[ USER-PROVIDED HELPER FUNCTIONS ]----------------------------

def extract_section(file_lines, start_marker, end_marker):
    """Extracts lines between the last occurrence of start_marker and end_marker."""
    pos_Start=[None]; pos_End=[None]
    for i, line in enumerate(file_lines):
        if start_marker in line: pos_Start.append(i)
        if end_marker in line: pos_End.append(i)
    if pos_Start[-1] is not None:
        # Check if we found a valid range
        if pos_End[-1] is not None and pos_End[-1] > pos_Start[-1]:
            return file_lines[pos_Start[-1] + 1 : pos_End[-1]]
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


def modify_gamma(lines, gamma_energy, new_energy, new_intensity):
    # Find gamma lines and modify specified gamma
    new_energy2 = [float(item) for item in new_energy]
    new_intensity2 = [float(item) for item in new_intensity]
    num = 0
    modified = False
    for i, line in enumerate(lines):
        if(num<=9):
            num = num + 1
            if line.startswith('     '):  # Identifies a gamma line based on the format of spaces given before the gamma.
                parts = line.split()
                for m in range(len(gamma_energy)):
                    if float(parts[1]) == round(gamma_energy[m]*1000,2):
                        if new_energy2 is not None:
                            parts[1] = f"{new_energy2[m]:.3f}"  # Update energy
                        if new_intensity2 is not None:
                            parts[7] = f"{new_intensity2[m]:.3f}"  # Update intensity
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
                        if new_energy2 is not None:
                            parts[1] = f"{new_energy2[m]:.3f}"  # Update energy
                        if new_intensity2 is not None:
                            parts[7] = f"{new_intensity2[m]:.3f}"  # Update intensity
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


def execute_gls_conversion(ags_file, gls_output_file):
    """
    Executes gls_conv and gls using subprocess.
    The GOSIA executables must be available in the system's PATH.
    """
    # Input data required by gls_conv
    gls_conv_input = f"4\n{ags_file}\n{gls_output_file}\ny\n"
    
    try:
        # 1. Execute gls_conv: Convert AGS file to GLS format
        result_conv = subprocess.run(
            ['gls_conv'], 
            input=gls_conv_input, 
            capture_output=True, 
            text=True, 
            timeout=10,
            check=True # Raise CalledProcessError for non-zero exit codes
        )
        
        # 2. Execute gls: Launch the Level Scheme Viewer
        # NOTE: This command is often blocking and launches a separate GUI window.
        # It's included as requested, but might behave differently depending on your setup.
        subprocess.Popen(['gls', gls_output_file]) 
        
        return True, "GLS file was edited and converted from AGS to GLS, and the 'gls' viewer was launched."

    except FileNotFoundError:
        return False, "GOSIA executable 'gls_conv' or 'gls' not found. Ensure GOSIA executables are in your system's PATH."
    except subprocess.CalledProcessError as e:
        return False, f"GOSIA utility 'gls_conv' failed (Code {e.returncode}). Please check GOSIA files and permissions.\nError output: {e.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "GOSIA process timed out. It might be hanging or taking too long."
    except Exception as e:
        return False, f"An unknown error occurred during subprocess execution: {e}"

# ----------------------------------------------------------------------------------------

def save(MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data,run_num = Minimization_Run_Num):
    counter = 0
    filename = './CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'
    while os.path.isfile(filename.format(run_num,counter)):
        counter = counter + 1
    filepath = Path('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter))  
    filepath.parent.mkdir(parents=True, exist_ok=True) 

    save_args = {'sep':'\t', 'na_rep':' ', 'index':False, 'header':False, 'encoding':'utf-8', 'quoting':csv.QUOTE_ALL}
    
    MATRIX_Data.dropna(axis=1, how='all').to_csv('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num, counter), **save_args)
    Lifetime_Data.dropna(axis=1, how='all').to_csv('./CSV_OUTPUT/Lifetime_Data_MINIRUN_{}_{}.csv'.format(run_num, counter), **save_args)
    MixingRatio_Data.dropna(axis=1, how='all').to_csv('./CSV_OUTPUT/MixingRatio_Data_MINIRUN_{}_{}.csv'.format(run_num, counter), **save_args)
    Yields_Data.dropna(axis=1, how='all').to_csv('./CSV_OUTPUT/Yields_Data_MINIRUN_{}_{}.csv'.format(run_num, counter), **save_args)
    BR_Data.dropna(axis=1, how='all').to_csv('./CSV_OUTPUT/BR_Data_MINIRUN_{}_{}.csv'.format(run_num, counter), **save_args)
    return counter

def display_prev(value,run_num = Minimization_Run_Num):
    try:
        counter = value
        MATRIX_Data=pd.read_csv('./CSV_OUTPUT/Matrix_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        Lifetime_Data=pd.read_csv('./CSV_OUTPUT/Lifetime_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        MixingRatio_Data=pd.read_csv('./CSV_OUTPUT/MixingRatio_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        Yields_Data=pd.read_csv('./CSV_OUTPUT/Yields_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        BR_Data=pd.read_csv('./CSV_OUTPUT/BR_Data_MINIRUN_{}_{}.csv'.format(run_num,counter),sep='\t')
        return MATRIX_Data,Lifetime_Data,MixingRatio_Data,Yields_Data,BR_Data
    except Exception:
        messagebox.showerror("Error", "Could not load data for run number {}. Check file existence and format.".format(value))
        return None, None, None, None, None


def Read_out_file(file_path):
    try:
        with open(file_path, 'r', encoding='latin-1') as file:
            file_content = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            file_content = file.readlines()
            
    recorded_sections = {
        "Yields":{"start": "CALCULATED AND EXPERIMENTAL YIELDS", "end": "EXP. AND CALCULATED BRANCHING RATIOS"},
        "BR":{"start": "EXP. AND CALCULATED BRANCHING RATIOS", "end": "E2/M1 MIXING RATIOS"},
        "E2/M1 MIXING RATIOS": {"start": "E2/M1 MIXING RATIOS", "end": "CALCULATED LIFETIMES"},
        "Lifetimes":{"start": "CALCULATED LIFETIMES", "end": "CALCULATED AND EXPERIMENTAL MATRIX ELEMENTS"},
        "ME_Cal":{"start": "CALCULATED AND EXPERIMENTAL MATRIX ELEMENTS", "end": "MATRIX ELEMENTS"},
        "MATRIX ELEMENTS":{"start": "MATRIX ELEMENTS", "end": "********* END OF EXECUTION **********"}
    }
    
    # --- LEVELS Parsing ---
    level_header_line = -1
    matrix_elements_start_line = -1
    for k, line in enumerate(file_content):
        if "ENERGY(MEV)" in line: level_header_line = k
        if "MATRIX ELEMENTS" in line and k > level_header_line and level_header_line != -1:
            matrix_elements_start_line = k; break
            
    if level_header_line != -1 and matrix_elements_start_line != -1:
        data_start_line = level_header_line + 1
        level_lines = file_content[data_start_line:matrix_elements_start_line]; level_string = "".join(level_lines)
        LEVELS_DF = pd.read_csv(io.StringIO(level_string), sep='\s+', header=None, engine='python').dropna(how='all') 
        
        LEVELS_DF.columns = ['INDEX', 'PARITY', 'SPIN', 'ENERGY(MEV)']
        LEVELS_DF['INDEX'] = LEVELS_DF['INDEX'].astype(int)
        LEVELS_DF = LEVELS_DF.set_index('INDEX')
        Level_series = LEVELS_DF['ENERGY(MEV)']; Spin_series = LEVELS_DF['SPIN']
    else:
        # Crucial: Ensure the return values are ALWAYS pd.Series on failure
        Level_series = pd.Series(dtype=float); Spin_series = pd.Series(dtype=float)
    
    # --- Section Extraction and Cleaning ---
    parsed_data_with_skipping = {}
    for section, markers in recorded_sections.items(): 
        extracted_lines = extract_section(file_content, markers["start"], markers["end"])
        parsed_data_with_skipping[section] = extracted_lines 

    cleaned_data = {}; CleanedList=[]
    CLEAN_PATTERN = re.compile(r'[^\w\s\.\,\-\+\=\*\:]') 
    
    for section, lines in parsed_data_with_skipping.items():
        for i in lines:
            clean_line = CLEAN_PATTERN.sub('', i); j = clean_line.strip()
            
            if(len(j)>0 and not any(header in j for header in ['INDEX', 'NF', 'NS', 'ME'])):
                string = j.split()
                # DOUBLET SYSTEM CLEANUP FIX
                if len(string) == 13 or len(string) == 14:
                    ni_col = string[0] + string[1] + string[2] + string[3]
                    ii_col = string[4] + string[5] + string[6] + string[7]
                    energy_split = string[8].split('+')
                    if_col = energy_split[0]
                    energy_col = energy_split[1] if len(energy_split) > 1 else 'NULL'
                    
                    string = [
                        ni_col, 'NULL', ii_col, if_col, energy_col, string[9], string[10], string[11], string[12]
                    ]

                if len(string) == 12:

                    ni_col = string[0] + string[1] + string[2]
                    ii_col = string[3] +string[4] + string[5] + string[6] 
                    energy_split = string[7].split('+')
                    if_col = energy_split[0]
                    energy_col = energy_split[1] if len(energy_split) > 1 else 'NULL'
                    
                    string = [
                        ni_col, 'NULL', ii_col, if_col, energy_col, string[8], string[9], string[10], string[11]
                    ]

                if (len(string) == 10 and (string[9] != None and string[9] != '**')):
                    ni_col = string[0]
                    ii_col = string[1] +string[2] + string[3] + string[4] 
                    energy_split = string[5].split('+')
                    if_col = energy_split[0]
                    energy_col = energy_split[1] if len(energy_split) > 1 else 'NULL'
                    
                    string = [
                        ni_col, 'NULL', ii_col, if_col, energy_col, string[6], string[7], string[8], string[9]
                    ]

                CleanedList.append(string)

        cleaned_data[section]=CleanedList.copy(); CleanedList.clear()

    Yields_Data=pd.DataFrame.from_dict(cleaned_data["Yields"])
    BR_Data=pd.DataFrame.from_dict(cleaned_data["BR"])
    MixingRatio_Data=pd.DataFrame.from_dict(cleaned_data["E2/M1 MIXING RATIOS"])
    Lifetime_Data=pd.DataFrame.from_dict(cleaned_data["Lifetimes"])
    MATRIX_Data=pd.DataFrame.from_dict(cleaned_data["ME_Cal"])
    return cleaned_data,Yields_Data,BR_Data,MixingRatio_Data,Lifetime_Data,MATRIX_Data, Level_series, Spin_series

def Matrix_2_Transitions(cleaned_data, Level_series, Spin_series):
    Energy = []; M_Ele_E2 = []; M_Ele_M1 = []; Norm_BE2 = []; Q_vals = []; Qt_vals = {}
    Status = False; StatusE3 = False; StatusE1 = False
    GS_BAND_T = {}
    Intr_BAND_T = {}
    Gam_BAND_T = {}

    for i,x in enumerate(GSBand):
        GS_BAND_T[x] = [0., 0.,0] # Transitional ME, Diagonal ME
        Qt_vals[x] = [0,0]
    for i,x in enumerate(IntrBand):
        Intr_BAND_T[x] = [0., 0.,0]
        Qt_vals[x] = [0,0]
    for i,x in enumerate(GamBand):
        Gam_BAND_T[x] = [0., 0.,0]
        Qt_vals[x] = [0,0]
    if not isinstance(Level_series, pd.Series) or not isinstance(Spin_series, pd.Series):
        messagebox.showwarning("Data Warning", "Level data parsing failed: Level or Spin information is missing or corrupted.")
        return [], [], [], [], [], []
        
    if Level_series.empty or Spin_series.empty:
        return [], [], [], [], [], []
    # -----------------------------------------------------------------
    
    for Col in cleaned_data["MATRIX ELEMENTS"]:
        if Col[0] == 'MULTIPOLARITY=2': Status = True; StatusE3 = False; StatusE1 = False
        if Col[0] == 'MULTIPOLARITY=7': Status = False; StatusE3 = False; StatusE1 = False
        if Col[0] == 'MULTIPOLARITY=1': Status = False; StatusE3 = False; StatusE1 = True
        if Col[0] == 'MULTIPOLARITY=3': Status = False; StatusE3 = True; StatusE1 = False

        if len(Col) > 1 and len(Col) < 9 and 'INDEX' not in Col: 
            try:
                start_level_index = int(Col[2]); final_level_index = int(Col[1])
                # Accessing .loc is safe now because of the initial check
                Lv1 = Level_series.loc[start_level_index]; Lv2 = Level_series.loc[final_level_index]      
                
                # Spin lookups are also safe
                if(Spin_series.loc[final_level_index] > Spin_series.loc[start_level_index]):
                    Spin_up = Spin_series.loc[final_level_index]; Spin_down = Spin_series.loc[start_level_index]      
                else:
                    Spin_up = Spin_series.loc[start_level_index]; Spin_down = Spin_series.loc[final_level_index]     
            except KeyError:
                continue
            
            if(Status): # MULTIPOLARITY=2
                Energy.append(float(abs(Lv1-Lv2))) 
                # Logic for Q-values
                if(abs(Lv1-Lv2)==0):
                    # Ground state band
                    if(Lv1*1000 in GSBand):
                        Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),0),Spin_up,float(Col[3])))
                        GS_BAND_T[Lv1*1000][1] = float(Col[3])
                        GS_BAND_T[Lv1*1000][2] = int(Spin_up)
                    if(Lv1*1000 in GamBand):
                        if(int(Spin_up) == 2): continue
                        Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),2),Spin_up,float(Col[3])))
                        Gam_BAND_T[Lv1*1000][1] = float(Col[3])
                        Gam_BAND_T[Lv1*1000][2] = int(Spin_up)
                    if(Lv1*1000 in IntrBand): 
                        Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),2),Spin_up,float(Col[3])))
                        Intr_BAND_T[Lv1*1000][1] = float(Col[3])
                        Intr_BAND_T[Lv1*1000][2] = int(Spin_up)

                    elif(Lv1*1000 in K4Band): 
                        Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),4),Spin_up,float(Col[3])))    
                    else: 
                        Q_vals.append((Lv1,ME_Q_0(float(Col[3]),int(Spin_up),0),Spin_up,float(Col[3])))

                # Transitions between the in-band levels
                GS_T = [b - a for a, b in zip(GSBand, GSBand[1:])]
                GmB_T = [b - a for a, b in zip(GamBand, GamBand[1:])]
                InB_T = [b - a for a, b in zip(IntrBand, IntrBand[1:])]

                m = int(max(Lv1*1000, Lv2*1000))

                if(round(float(abs(Lv1-Lv2)*1000)) in GS_T):
                    GS_BAND_T[m][0] = float(Col[3])
                if(round(float(abs(Lv1-Lv2)*1000)) in GmB_T): 
                    Gam_BAND_T[m][0] = float(Col[3])
                if(round(float(abs(Lv1-Lv2)*1000)) in InB_T):
                    Intr_BAND_T[m][0] = float(Col[3])

                M_Ele_E2.append(ME_2_Wu(float(Col[3]),2.0,Spin_up,Spin_down)); M_Ele_M1.append(0)


            elif(StatusE3): # MULTIPOLARITY=3
                Energy.append(float(abs(Lv1-Lv2))) 
                M_Ele_E2.append(ME_2_Wu(float(Col[3]),3.0,Spin_up,Spin_down))

            elif(StatusE1): # MULTIPOLARITY=1
                Energy.append(float(abs(Lv1-Lv2))) 
                M_Ele_E2.append(ME_2_Wu(float(Col[3]),1.0,Spin_up,Spin_down))

    
    # Q value Ratios
    for Lvl in GSBand:
        if(GS_BAND_T[Lvl][0] == 0.0): continue
        Qt_vals[Lvl] = [GS_BAND_T[Lvl][2],ME_Q_ratios(GS_BAND_T[Lvl][0],GS_BAND_T[Lvl][1],GS_BAND_T[Lvl][2])]
    for Lvl in GamBand:
        if(Gam_BAND_T[Lvl][0] == 0.0): continue
        Qt_vals[Lvl] = [Gam_BAND_T[Lvl][2],ME_Q_ratios2(Gam_BAND_T[Lvl][0],Gam_BAND_T[Lvl][1],Gam_BAND_T[Lvl][2])]
    for Lvl in IntrBand:
        if(Intr_BAND_T[Lvl][0] == 0.0): continue
        Qt_vals[Lvl] = [Intr_BAND_T[Lvl][2],ME_Q_ratios(Intr_BAND_T[Lvl][0],Intr_BAND_T[Lvl][1],Intr_BAND_T[Lvl][2])]


    return Energy,M_Ele_E2,M_Ele_M1,Norm_BE2,Q_vals,Qt_vals

def update_ags_file(ags_file_path, Energy, M_Ele_E2, Q_vals):
    """
    Reads an existing AGS file, updates it with calculated B(E2) values and Q-values,
    saves the modified content to 'items.ags', and then converts it to GLS format.
    """

    with open(ags_file_path, 'r') as file:
        lines = file.readlines()

    recorded_sections ={
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


def visualize_data_plots(cleaned_data, Cal_counter, plot_flags):
    """
    Generates Matplotlib figures based on selected plot flags.
    Fixes label overlap, and uses dynamic sizing for better display.
    """
    # 0. Configuration
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Enable Matplotlib's default zoom/pan tools by using an interactive backend
    # Note: The *user* must be running this code in a suitable environment (e.g., IPython with an interactive backend like Qt or Tk)
    # The default behavior of a figure displayed in a window will automatically include zoom/pan tools.
    
    plot_map = {
        'lifetime': {'title': 'GOSIA: Lifetimes', 'data_key': 'Lifetime', 'y_label': 'Lifetime (ps)', 'x_label': 'Level Index'},
        'br': {'title': 'GOSIA: Branching Ratios', 'data_key': 'BR', 'y_label': 'Branching Ratio', 'x_label': 'Transition Index'},
        'yields': {'title': 'GOSIA: Yields', 'data_key': 'Yields', 'y_label': 'Yield Difference', 'x_label': 'Transition Energy (keV)'},
        'mixing': {'title': 'GOSIA: E2/M1 Mixing Ratios', 'data_key': 'MixingRatio', 'y_label': 'Mixing Ratio $\delta(E2/M1)$', 'x_label': 'Transition Index'},
        'Matrix': {'title': 'GOSIA: Given Matrix Elements (CALCULATED VS. EXPERIMENTAL)', 'data_key': 'Matrix', 'y_label': 'Matrix Elements (eb)', 'x_label': 'Energy (keV)'},
        'QuadrupoleK0': {'title': 'GOSIA: K=0 Spectroscopic Q Value [in units of Q0]', 'data_key': 'QuadrupoleK0', 'y_label': 'Qs/Q0', 'x_label': 'Spin(I)'},
        'QuadrupoleK2': {'title': 'GOSIA: K=2 Spectroscopic Q Value [in units of Q0]', 'data_key': 'QuadrupoleK2', 'y_label': 'Qs/Q0', 'x_label': 'Spin(I)'}
    }

    selected_plots = [key for key, is_selected in plot_flags.items() if is_selected]
    num_plots = len(selected_plots)
    
    # Handle no plots selected
    if num_plots == 0:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No main plots selected.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # 1. Determine subplot layout (Max 2 rows)
    if num_plots == 1:
        nrows, ncols = 1, 1
    elif num_plots == 2:
        nrows, ncols = 1, 2
    elif num_plots == 3:
        nrows, ncols = 2, 2
    elif num_plots >= 4:
        nrows, ncols = 2, 2

    # Dynamic Figure Size: Scale size based on number of plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows)) 
    axes_list = np.array(axes).flatten()
    
    shades = cyan_shades(Cal_counter + 1)

    # 2. Start Plotting
    for i, plot_key in enumerate(selected_plots):
        ax = axes_list[i]
        plot_info = plot_map[plot_key]

        ax.set_title(plot_info['title'], fontsize=12)
        ax.set_xlabel(plot_info['x_label'])
        ax.set_ylabel(plot_info['y_label'])
        
        # --- Plotting Logic based ONLY on plot_key ---
        
        # Plot: Yields
        if plot_key == 'yields':
            GY_Label = []
            GY_Cal = []
            GY_Exp = []
            GY_ERR = []
            rows = cleaned_data['Yields'].shape[0]
            for row in range(0, rows):
                if cleaned_data['Yields'].iloc[row, 8] is not None:
                    if(cleaned_data['Yields'].iloc[row, 8] == '****'):
                        cleaned_data['Yields'].iloc[row, 8] = 10.0

                    cal = (float(cleaned_data['Yields'].iloc[row, 8])) # SIGMA VALUE RN
                    cal_Yield = float(cleaned_data['Yields'].iloc[row, 5])
                    exp = float(cleaned_data['Yields'].iloc[row, 6])

                    # Check for doublet yields
                    if(abs(float(cleaned_data['Yields'].iloc[row, 8])) != 0):
                        Err_cal=cal
                    else:
                        Err_cal= 0 

                    if(cal >= 1.0 or cal <= 1.0):
                        if cleaned_data['Yields'].iloc[row, 1] == "NULL":

                            energy1 = float(cleaned_data['Yields'].iloc[row, 4]) * 1000
                            energy2 = float(cleaned_data['Yields'].iloc[row, 3]) * 1000
                            GY_Label.append(int(energy1))
                            GY_Cal.append(cal_Yield)
                            GY_Exp.append(exp)
                            GY_ERR.append(Err_cal)
                            GY_Label.append(int(energy2))
                            GY_Cal.append(cal_Yield)
                            GY_Exp.append(exp)
                            GY_ERR.append(Err_cal)

                        else:

                            energy = float(cleaned_data['Yields'].iloc[row, 4]) * 1000
                            GY_Label.append(int(energy))
                            GY_Cal.append(cal_Yield)
                            GY_Exp.append(exp)
                            GY_ERR.append(Err_cal)

            paired_lists = list(zip(GY_Label, GY_ERR))

            sorted_pairs = sorted(paired_lists, key=lambda pair: pair[0])

            if sorted_pairs:
                sorted_list1, sorted_list2 = zip(*sorted_pairs)
                sorted_list1 = list(map(str, sorted_list1))
            else:
                sorted_list1 = []
                sorted_list2 = []


            ax.bar(sorted_list1, sorted_list2,width=0.6, edgecolor='red',facecolor='none',alpha=1.0,hatch='////',label= "*Recent Calc*");


            # FIX: Rotate x-axis labels for Yields
            ax.tick_params(axis='x', rotation=45)
            
        # Plot: Mixing Ratios
        elif plot_key == 'mixing':
            MIX_Label = []
            MIX_EXP = []
            MIX_Cal = []
            MIX_sig = []
            rows = cleaned_data['MixingRatio'].shape[0]

            for row in range(0, rows):
                MIX_Label.append(str(cleaned_data['MixingRatio'].iloc[row, 0]) + '-->' + str(cleaned_data['MixingRatio'].iloc[row, 1]))
                MIX_EXP.append(float(cleaned_data['MixingRatio'].iloc[row, 3]))
                MIX_Cal.append(float(cleaned_data['MixingRatio'].iloc[row, 2]))
                
            ax.scatter(
                MIX_Label, 
                MIX_EXP,  
                color='#007ACC',
                label='Experimental'
            )

            ax.scatter(
                MIX_Label, 
                MIX_Cal, 
                color='#FF4B4B',
                label='Calculated'
            )
            ax.legend()
            # FIX: Rotate x-axis labels for Mixing Ratios
            ax.tick_params(axis='x', rotation=45)
            
        # Plot: Branching Ratios
        elif plot_key == 'br':
            Index = []
            LT_EXP = []
            LT_Cal = []
            LT_sig = []
            rows = cleaned_data['BR'].shape[0]
            for row in range(0, rows):
                Index.append(str(cleaned_data['BR'].iloc[row, 0]) + '-->' + str(cleaned_data['BR'].iloc[row, 1]))
                LT_EXP.append(float(cleaned_data['BR'].iloc[row, 4]))
                LT_Cal.append(float(cleaned_data['BR'].iloc[row, 7]))
                LT_sig.append(abs(float(cleaned_data['BR'].iloc[row, 5])))

            LT_EXP_array = np.array(LT_EXP)
            LT_sig_array = np.array(LT_sig)
            
            # Calculate the upper and lower bounds for the sigma band
            LT_Upper = LT_EXP_array + LT_sig_array
            LT_Lower = LT_EXP_array - LT_sig_array
            ax.fill_between(
                Index,
                LT_Lower,
                LT_Upper,
                color='cornflowerblue', 
                alpha=0.3, # Transparency level
                label='Experimental $\pm 1\sigma$'
            )

            ax.errorbar(
                Index, 
                LT_EXP, 
                yerr=LT_sig,
                fmt='o', 
                color='#007ACC',
                capsize=4,
                label='Experimental'
            )
            ax.scatter(
                Index, 
                LT_Cal, 
                color='#FF4B4B',
                label='Calculated'
            )
            ax.legend()
            # FIX: Rotate x-axis labels for Branching Ratios
            ax.tick_params(axis='x', rotation=45)
            
        # Plot: Lifetimes
        elif plot_key == 'lifetime':
            Index = []
            LT_EXP = []
            LT_Cal = []
            LT_sig = []
            rows = cleaned_data['Lifetime'].shape[0]
            for row in range(0, rows):
                if cleaned_data['Lifetime'].iloc[row, 3] is not None:
                    Index.append(int(cleaned_data['Lifetime'].iloc[row, 0]))
                    LT_EXP.append(float(cleaned_data['Lifetime'].iloc[row, 2]))
                    LT_Cal.append(float(cleaned_data['Lifetime'].iloc[row, 1]))
                    sig = (float(cleaned_data['Lifetime'].iloc[row, 3]))
                    LT_sig.append(abs(sig))
                    
            ax.errorbar(
                Index, 
                LT_EXP, 
                yerr=LT_sig,
                fmt='o', 
                color='#007ACC',
                capsize=4,
                label='Experimental'
            )
            ax.scatter(
                Index, 
                LT_Cal, 
                color='#FF4B4B',
                label='Calculated'
            )
            ax.legend()

        elif plot_key == 'Matrix':
            ME_Label= []
            ME_Exp = []
            ME_Err = []
            ME_Cal = []
            rows = cleaned_data['Matrix'].shape[0]
            print(cleaned_data['Matrix'])
            for row in range(0, rows):
                if(abs(float(cleaned_data['Matrix'].iloc[row, 4])) != 0):
                    Err_cal=abs(float(cleaned_data['Matrix'].iloc[row, 3])-float(cleaned_data['Matrix'].iloc[row, 2]))/abs(float(cleaned_data['Matrix'].iloc[row, 4]))
                else:
                    Err_cal= 0 

                if(abs(float(cleaned_data['Matrix'].iloc[row, 4]))>=1.0):
                    ME_Label.append(cleaned_data['Matrix'].iloc[row, 0]+'-->'+cleaned_data['Matrix'].iloc[row, 1])
                    ME_Cal.append(float(cleaned_data['Matrix'].iloc[row, 3]))
                    ME_Exp.append(float(cleaned_data['Matrix'].iloc[row, 2]))
                    ME_Err.append(Err_cal)                    
           
            ax.errorbar(
                ME_Label, 
                ME_Exp, 
                yerr=ME_Err,
                fmt='o', 
                color='#007ACC',
                capsize=4,
                label='Experimental'
            )

            ax.scatter(
                ME_Label, 
                ME_Cal, 
                color='#FF4B4B',
                label='Gosia Evaluated'
            )
            ax.legend()
            ax.tick_params(axis='x', rotation=45)


        # ====================================================================
        # QUADRUPOLE K=2 PLOT LOGIC
        # ====================================================================
        if plot_key == 'QuadrupoleK2':
            ax.set_title(r'Quadrupole Moment Ratios for $K=2$ Band', fontsize=14)

            # --- Plot Theoretical Lines ---
            ax.plot(I_k2, q_k2_gamma_10, marker='o', linestyle='-', color=gamma_colors[10],
                    linewidth=2, markersize=7, fillstyle='none')
            ax.plot(I_k2, q_k2_gamma_20, marker='s', linestyle='-', color=gamma_colors[20],
                    linewidth=2, markersize=7, fillstyle='none')
            ax.plot(I_k2, q_k2_gamma_27_5, marker='^', linestyle='-', color=gamma_colors[27.5],
                    linewidth=2, markersize=7, fillstyle='none')
            ax.plot(I_k2, q_calculated_K2, marker='X', linestyle=':', color=model_colors['K2'],
                    linewidth=2.5, markersize=7)

            # --- Add Annotations (Corrected and Robust) ---
            last_idx_k2 = len(I_k2) - 1 # Last valid index is 8
            
            # K=2, gamma=10
            ax.annotate(r'$\gamma=10^\circ$', xy=(I_k2[last_idx_k2], q_k2_gamma_10[last_idx_k2]),
                        xytext=(I_k2[last_idx_k2] + 0.1, q_k2_gamma_10[last_idx_k2]),
                        color=gamma_colors[10], fontsize=10)
            
            # K=2, gamma=27.5
            ax.annotate(r'$\gamma=27.5^\circ$', xy=(I_k2[4], q_k2_gamma_27_5[4]), # Placed at index 4
                        xytext=(I_k2[4] + 0.5, q_k2_gamma_27_5[4]),
                        color=gamma_colors[27.5], fontsize=10,
                        arrowprops=dict(facecolor=gamma_colors[27.5], shrink=0.05, width=0.5, headwidth=5))

            ax.annotate(r'$K=2, \gamma=20^\circ$', xy=(I_k2[last_idx_k2], q_k2_gamma_20[last_idx_k2]),
                        xytext=(I_k2[last_idx_k2] + 0.2, q_k2_gamma_20[last_idx_k2] - 0.01),
                        color=gamma_colors[20], fontsize=10)

            # ROTOR Model K=2
            ax.annotate(r'ROTOR Model $K=2$', xy=(I_k2[last_idx_k2], q_calculated_K2[last_idx_k2]),
                        xytext=(I_k2[last_idx_k2] + 0.5, q_calculated_K2[last_idx_k2] + 0.05),
                        color=model_colors['K2'], fontsize=10,
                        arrowprops=dict(facecolor=model_colors['K2'], shrink=0.05, width=0.5, headwidth=5))

            spin = []
            ratio = []
            for Lvl, val in cleaned_data['Q_ratios'].items():
                if Lvl in GamBand:
                    spin.append(val[0])
                    ratio.append(val[1])

            if spin and ratio:
                ax.scatter(spin, ratio,
                           color=experimental_color,
                           marker='D',
                           s=100,
                           edgecolor='black',
                           linewidth=1.5,
                           zorder=5)

            ax.set_xlim(1.5, 12.5)

        # ====================================================================
        # QUADRUPOLE K=0 PLOT LOGIC
        # ====================================================================
        elif plot_key == 'QuadrupoleK0':
            ax.set_title(r'Quadrupole Moment Ratios for $K=0$ Band', fontsize=14)

            # --- Plot Theoretical Lines ---
            ax.plot(I_k0, q_k0_gamma_10, marker='o', linestyle='--', color=gamma_colors[10],
                    linewidth=2, markersize=7)
            ax.plot(I_k0, q_k0_gamma_20, marker='s', linestyle='--', color=gamma_colors[20],
                    linewidth=2, markersize=7)
            ax.plot(I_k0, q_k0_gamma_27_5, marker='^', linestyle='--', color=gamma_colors[27.5],
                    linewidth=2, markersize=7)
            ax.plot(I_k0, q_calculated_K0, marker='D', linestyle='-.', color=model_colors['K0'],
                    linewidth=2.5, markersize=6)

            # --- Add Annotations ---
            last_idx_k0 = len(I_k0) - 1 # Last valid index is 4

            # K=0, gamma=27.5
            ax.annotate(r'$K=0, \gamma=27.5^\circ$', xy=(I_k0[last_idx_k0], q_k0_gamma_27_5[last_idx_k0]),
                        xytext=(I_k0[last_idx_k0] + 0.2, q_k0_gamma_27_5[last_idx_k0]),
                        color=gamma_colors[27.5], fontsize=10)
            
            # K=0, gamma=10
            ax.annotate(r'$K=0, \gamma=10^\circ$', xy=(I_k0[last_idx_k0], q_k0_gamma_10[last_idx_k0]),
                        xytext=(I_k0[last_idx_k0] + 0.2, q_k0_gamma_10[last_idx_k0] - 0.01),
                        color=gamma_colors[10], fontsize=10)

            # K=0, gamma=20
            ax.annotate(r'$K=0, \gamma=20^\circ$', xy=(I_k0[last_idx_k0], q_k0_gamma_20[last_idx_k0]),
                        xytext=(I_k0[last_idx_k0] + 0.2, q_k0_gamma_20[last_idx_k0] - 0.01),
                        color=gamma_colors[20], fontsize=10)
            
            # ROTOR Model K=0
            ax.annotate(r'ROTOR Model $K=0$', xy=(I_k0[2], q_calculated_K0[2]), # Placed at index 2
                        xytext=(I_k0[2] + 0.5, q_calculated_K0[2]),
                        color=model_colors['K0'], fontsize=10,
                        arrowprops=dict(facecolor=model_colors['K0'], shrink=0.05, width=0.5, headwidth=5))

            # --- Collect and Plot Prominent Experimental Data ---
            spin = []
            ratio = []
            spin2 = []
            ratio2 = []
            for Lvl, val in cleaned_data['Q_ratios'].items():
                if Lvl in GSBand:
                    spin.append(val[0])
                    ratio.append(val[1])

            for Lvl, val in cleaned_data['Q_ratios'].items():
                if Lvl in IntrBand:
                    spin2.append(val[0])
                    ratio2.append(val[1])
                    
            if spin and ratio:
                ax.scatter(spin, ratio,
                           color=experimental_color,
                           marker='D',
                           s=100,
                           edgecolor='black',
                           linewidth=1.5,
                           label="G.S Band",
                           zorder=5)

            if spin2 and ratio2:
                ax.scatter(spin2, ratio2,
                           color='purple',
                           marker='s',
                           s=100,
                           edgecolor='black',
                           linewidth=1.5,
                           label="IntrBand",
                           zorder=5)
            ax.legend()
            ax.set_xlim(1.5, 12.5)


    # 3. Remove unused axes if num_plots is 3
    if 1 < num_plots <= 3:
        for j in range(num_plots, nrows * ncols):
            fig.delaxes(axes_list[j])
            
    # 4. Finalize layout
    # The pad= option adds extra space around the figure, further preventing overlap
    fig.tight_layout(pad=3.0)
    
    return fig


# ==============================================================================
# TKINTER APPLICATION CLASS
# ==============================================================================

class GOSIA_GUI:
    def __init__(self, master):
        self.master = master
        master.title("GOSIA Data Analysis Tool")

        self.file_path = 'partial_band.out'
        # Default path for the AGS file (used in the user's provided logic)
        self.ags_file_path = '/home/sangeetpannu/Phd_110Cd_Project/110_Cd_PARTIAL.ags' 
        self.data_loaded = False
        self.data_frames = {}; self.series_data = {}
        self.analysis_results = {'Energy': [], 'M_Ele_E2': [], 'Q_vals': [], 'Q_ratios': []}

        # --- Frame Setup ---
        self.control_frame = tk.Frame(master, padx=10, pady=10); self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.plot_frame = tk.Frame(master); self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.current_canvas = None 

        # --- Plot Selection Checkboxes (FIXED with Grid Manager) ---
        # Create a control frame for the plot selection elements
        plot_control_frame = tk.Frame(master, padx=10, pady=5)
        # Use pack for the frame itself to position it within the main window
        plot_control_frame.pack(side=tk.TOP, fill=tk.X)

        # Label for the section
        # Row 0, Column 0
        tk.Label(plot_control_frame, text="Select Plots to Display:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        # Row 0: Lifetimes, BR, Yields
        self.plot_lifetime_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Lifetimes", variable=self.plot_lifetime_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        self.plot_br_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Branching Ratios", variable=self.plot_br_var).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        self.plot_yields_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Yields", variable=self.plot_yields_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Row 1: Mixing Ratios, Matrix Elements, Quadrupole K=0
        self.plot_mixing_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Mixing Ratios", variable=self.plot_mixing_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.plot_ME_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Matrix Elements", variable=self.plot_ME_var).grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # K=0 Plot
        self.plot_QE_varK0 = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_control_frame, text="Quadrupole Ratios (K=0)", variable=self.plot_QE_varK0).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        # Row 2: Quadrupole K=2 (Now on its own line for better separation)
        self.plot_QE_varK2 = tk.BooleanVar(value=False)
        # This moves to the next row (row=2) to prevent overlap
        tk.Checkbutton(plot_control_frame, text="Quadrupole Ratios (K=2)", variable=self.plot_QE_varK2).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        # --- Buttons ---
        button_frame = tk.Frame(master, padx=10, pady=5)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(button_frame, text="Draw Plots", command=self.draw_plot).pack(side=tk.LEFT, padx=5)

        # --- Plotting Frame ---
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # --- UI Elements ---
        tk.Label(self.control_frame, text="Output File:").pack(pady=5)
        self.file_label = tk.Label(self.control_frame, text=os.path.basename(self.file_path), fg="blue"); self.file_label.pack()
        self.select_button = tk.Button(self.control_frame, text="Select .out File", command=self.select_file); self.select_button.pack(pady=5)
        
        tk.Label(self.control_frame, text="AGS File Path:").pack(pady=5)
        self.ags_label = tk.Label(self.control_frame, text=os.path.basename(self.ags_file_path), fg="green"); self.ags_label.pack()
        tk.Button(self.control_frame, text="Set .ags File", command=self.select_ags_file).pack(pady=5)

        self.read_button = tk.Button(self.control_frame, text="Read, Analyze", command=self.read_and_analyze, bg="lightgreen"); self.read_button.pack(pady=10)

        self.read_button = tk.Button(self.control_frame, text="Display gls", command=self.display_gls, bg="lightgreen"); self.read_button.pack(pady=10)
       
        tk.Frame(self.control_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)
        self.save_button = tk.Button(self.control_frame, text="Save Raw Data to CSV", command=self.save_data, state=tk.DISABLED, bg="lightblue"); self.save_button.pack(pady=5)
        
        tk.Label(self.control_frame, text="Load Previous Run:").pack(pady=5)
        self.run_entry = tk.Entry(self.control_frame, width=5); self.run_entry.insert(0, "0"); self.run_entry.pack(side=tk.LEFT, padx=5)
        self.display_button = tk.Button(self.control_frame, text="Load", command=self.display_previous_run); self.display_button.pack(side=tk.LEFT, padx=5)
        tk.Button(self.control_frame, text="Exit", command=master.quit, fg="red").pack(pady=20)
        

    def select_file(self):
        new_file = filedialog.askopenfilename(defaultextension=".out", filetypes=[("GOSIA Output Files", "*.out"), ("All Files", "*.*")])
        if new_file:
            self.file_path = new_file; self.file_label.config(text=os.path.basename(new_file))
            self.data_loaded = False; self.save_button.config(state=tk.DISABLED)
            #messagebox.showinfo("File Selected", f"Output file set to: {os.path.basename(self.file_path)}")

    def select_ags_file(self):
        new_file = filedialog.askopenfilename(defaultextension=".ags", filetypes=[("AGS Files", "*.ags"), ("All Files", "*.*")])
        if new_file:
            self.ags_file_path = new_file; self.ags_label.config(text=os.path.basename(new_file))
            #messagebox.showinfo("AGS File Selected", f"AGS file set to: {os.path.basename(self.ags_file_path)}")
      
    def display_gls(self):
        try:
            #messagebox.showinfo("Heeee not yet implemented")
            execute_gls_conversion("./items.ags", "test.gls")
            print(self.ags_file_path)
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {self.file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during reading/analysis/AGS update of '{os.path.basename(self.file_path)}':\n{e}")


    def read_and_analyze(self):
        try:
            # 1. Read and Parse Data
            results = Read_out_file(self.file_path)
            
            cleaned_data, self.data_frames['Yields'], self.data_frames['BR'], \
            self.data_frames['MixingRatio'], self.data_frames['Lifetime'], self.data_frames['Matrix'], \
            self.series_data['Level'], self.series_data['Spin'] = results

            # 2. Run Analysis
            Energy, M_Ele_E2, M_Ele_M1, Norm_BE2, Q_vals, Q_ratios = Matrix_2_Transitions(
                cleaned_data, self.series_data['Level'], self.series_data['Spin']
            )

            #for x in range(0,len(Energy)):
            #    print(f'Energy of {Energy[x]} has B(E2) of {M_Ele_E2[x]} W.u. \n')

            self.data_frames['Q_ratios'] = Q_ratios
            # Store Analysis Results
            self.analysis_results['Energy'] = Energy; self.analysis_results['M_Ele_E2'] = M_Ele_E2
            self.analysis_results['Q_vals'] = Q_vals; self.analysis_results['Q_ratios'] = Q_ratios
            
            # 3. Update AGS File and Execute GLS Conversion
            if self.ags_file_path:
                print("done")
                #update_ags_file(self.ags_file_path, Energy, M_Ele_E2, Q_vals)
            else:
                messagebox.showwarning("AGS Warning", "AGS file path is not set. Skipping AGS update and GLS conversion.")
                
            self.data_loaded = True; self.save_button.config(state=tk.NORMAL)

        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {self.file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during reading/analysis/AGS update of '{os.path.basename(self.file_path)}':\n{e}")

    def save_data(self):
        if not self.data_loaded:
            messagebox.showwarning("Warning", "No data loaded to save."); return

        save_counter = save(
            self.data_frames['Matrix'], self.data_frames['Lifetime'], self.data_frames['MixingRatio'],
            self.data_frames['Yields'], self.data_frames['BR']
        )
        messagebox.showinfo("Save Complete", f"Data successfully saved to CSV under run: MINIRUN_{Minimization_Run_Num}_{save_counter}")

    def display_previous_run(self):
        try:
            run_num = int(self.run_entry.get())
            MATRIX_Data, Lifetime_Data, MixingRatio_Data, Yields_Data, BR_Data = display_prev(run_num)
            if MATRIX_Data is None: return 
            messagebox.showinfo("Load Complete", f"Raw data for MINIRUN_{Minimization_Run_Num}_{run_num} loaded into memory. (No re-analysis/plot update).")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the run number.")
        
    def draw_plot(self):
        # Clear previous plot

        if not self.data_loaded:
            messagebox.showwarning("Plotting Error", "Please click 'Read, Analyze' first to load the data before attempting to plot.")
            return

        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        # 1. Get plot selection flags
        plot_flags = {
            'lifetime': self.plot_lifetime_var.get(),
            'br': self.plot_br_var.get(),
            'yields': self.plot_yields_var.get(),
            'mixing': self.plot_mixing_var.get(),
            'Matrix': self.plot_ME_var.get(),
            'QuadrupoleK0': self.plot_QE_varK0.get(),
            'QuadrupoleK2': self.plot_QE_varK2.get()
        }

        # Check if any plot is selected
        if not any(plot_flags.values()):
            messagebox.showinfo("No Selection", "Please select at least one plot to display.")
            return

        # 3. Generate the figure
        Cal_counter=1
        fig = visualize_data_plots(self.data_frames, Cal_counter, plot_flags)

        # 4. Embed the figure into the Tkinter window
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# ==============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = GOSIA_GUI(root)
    root.mainloop()