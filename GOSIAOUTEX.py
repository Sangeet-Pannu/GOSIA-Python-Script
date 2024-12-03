import pandas as pd
import math 
from pathlib import Path  
import datetime
import matplotlib.pyplot as plt
import numpy as np
import mplcursors  # Importing mplcursors for hover functionality 
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
    if L2 == 2:
        CleGoM = 0.534522484
    if L2 == 4:
        CleGoM = 0.713506068
    if L2 == 6:
        CleGoM = 0.792825
    if L2 == 3:
        CleGoM = 0.645497224
    if L2 == 5:
        CleGoM = 0.759555

    Qs = math.sqrt(16*math.pi/5)*CleGoM/(2*L2+1)*ME_Val
    Q0 = ((L2+1)*(2*L2+3)/(3*K**2-L2*(L2+1)))*Qs
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
#-----------------------------------------------------------------------------------------------|
#--------Function Definitions Above --------||

# Load the file
file_path = '110Cd.out'
i=0
with open(file_path, 'r') as file:
    file_content = file.readlines()



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

MATRIX_Data=pd.DataFrame.from_dict(cleaned_data["MATRIX ELEMENTS"])

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
            Spin_up = float(Spin_index[int(Col[2])])        # INITIAL LEVEL SPIN
            Lv2 = float(Level_index[int(Col[1])])           # FINAL LEVEL ENERGY FROM WHICH TRANSITION GOES TO
            Spin_down = float(Spin_index[int(Col[1])])      # FINAL LEVEL SPIN
            Energy.append(abs(Lv1-Lv2))
                                # We have the Energy of the transition
            if(abs(Lv1-Lv2)==0):
                Q_vals.append((Lv1,ME_Q_0(float(Col[3]),Spin_up,0),Spin_up))    # Q_value

            M_Ele_E2.append(ME_2_Wu(float(Col[3]),2.0,Spin_up,Spin_down))
            M_Ele_M1.append(0)
    if(Status==False):
        if len(Col) > 1 and len(Col) < 9:                    # makes sure we are not trying to grab lists that dont have more than 2 columns or have all string entries.
            Lv1 = float(Level_index[int(Col[2])])            # INITIAL LEVEL ENERGY FROM WHICH TRANSITION ORIGINATES
            Spin_up = float(Spin_index[int(Col[1])])         # INITIAL LEVEL SPIN
            Lv2 = float(Level_index[int(Col[1])])            # FINAL LEVEL ENERGY FROM WHICH TRANSITION GOES TO
            Spin_down = float(Spin_index[int(Col[2])])       # FINAL LEVEL SPIN
            indexM = Energy.index(abs(Lv1-Lv2))
            M_Ele_M1[indexM]=ME_1_Wu(float(Col[3]),1.0,Spin_up,Spin_down) # FIX THIS ISSUE!!! The index filled wont match the index of Energy for this M1 ME value!!!

# Prints the Transition Matrix Element in B(E2) Down [W.u]
for i in range(len(M_Ele_E2)):
    print("The Transition Energy: ",end="")
    print("{0:0.2f}".format(Energy[i]*1000),end="")
    print(" and The ME in B(E2) W.U: ",end="")
    print("{0:0.4f}".format(M_Ele_E2[i]))
    if(M_Ele_M1[i]>0):
        print(" and The ME in B(M1) W.U: ",end="")
        print("{0:0.4f}".format(M_Ele_M1[i]))

print("\n\n")
print("|-------------- Intrinsic Quadrupole Moments --------------|")
for x,y,z in Q_vals: 
    print("| Level Energy: {} | Spin = {} | Q_0 = {}".format(x*1000,z,y))
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

#--------[ Section: Data writing/storage to CSV file]--------------------------||

#filepath = Path('./CSV_OUTPUT/MATRIX_Data.csv')  
#filepath.parent.mkdir(parents=True, exist_ok=True) #Checks to see if the folder already exists
#MATRIX_Data.to_csv('./CSV_OUTPUT/MATRIX_Data.csv',index=False,header=False,mode='a')

#--------[ Section: Data writing/storage to CSV file]--------------------------||




#--------[ Section: Plotting Diagrams/Graphs ]--------------------------||

plt.style.use('dark_background')
plt.figure(figsize=(80, 60))
plt.subplot(2, 2, 1)
#----------[Lifetime Graphs]--------|
Index = []
LT_Exp = []
LT_Err = []
LT_Cal = []
for Col in cleaned_data["Lifetimes"]:
    if len(Col) <= 2:
        Index.append(float(Level_index[int(Col[0])])*1000)
        LT_Cal.append(float(Col[1]))
        LT_Exp.append(np.nan)
        LT_Err.append(0)
    if len(Col) > 4:
        Index.append(float(Level_index[int(Col[0])])*1000)
        LT_Cal.append(float(Col[1]))
        LT_Exp.append(float(Col[2]))
        LT_Err.append(abs(float(Col[3])))


errorbar_plot=plt.errorbar(Index, LT_Exp,yerr = LT_Err,label= "Experimental", color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );
scatter_plot=plt.scatter(Index, LT_Cal, label= "Calulated", color= "cyan",marker= "o", s=30)

# x-axis label
plt.xlabel('Level Energy (keV)')
plt.yscale("log")   
# frequency label
plt.ylabel('Lifetimes (Pico-Seconds)')
# plot title
plt.title('GOSIA: Lifetimes (CALCULATED VS. EXPERIMENTAL)')
# showing legend
plt.legend()

cursor = mplcursors.cursor([errorbar_plot, scatter_plot], hover=True)

# METHOD TO ALLOW HOVERING ANNOTATION.... 

@cursor.connect("add")
def on_add(sel):
    # Identify whether the point is experimental or calculated
    if sel.artist == errorbar_plot:
        sel.annotation.set_text(
            f"Exp:\nEnergy={sel.target[0]:.1f} keV\nLifetime={sel.target[1]:.4f} ps"
        )
    elif sel.artist == scatter_plot:
        sel.annotation.set_text(
            f"Calc:\nEnergy={sel.target[0]:.1f} keV\nLifetime={sel.target[1]:.4f} ps"
        )
    sel.annotation.arrow_patch.set_edgecolor("white")
    sel.annotation.arrow_patch.set_facecolor("white")

#---[Branching ratio Graphs]--------|
plt.subplot(2, 2, 2)

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

plt.errorbar(BR_Label, BR_Exp,yerr = BR_Err,label= "Experimental", color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );

plt.scatter(BR_Label, BR_Cal, label= "Calulated", color= "cyan",marker= "o", s=30)
# x-axis label
plt.xlabel('Li-->Lf/(Norm)')
#plt.yscale("log")   
# frequency label
plt.ylabel('Branching Ratio')
# plot title
plt.title('GOSIA: Branching Ratios (CALCULATED VS. EXPERIMENTAL)')
# showing legend
plt.legend()

#plt.show()

#------------[Yields Graphs]--------|
plt.subplot(2, 2, 3)
GY_Label= []
GY_Exp = []
GY_Err = []
GY_Cal = []
highlight_indices = []
num = 0
for Col in cleaned_data["Yields"]:
    num = num + 1
    if num > 1 and len(Col) > 4:
        GY_Label.append(float(Col[4])*1000)
        if(len(Col) == 6): 
            GY_Cal.append(float(Col[5]))
            GY_Exp.append(np.nan)
            GY_Err.append(np.nan)
        if(len(Col) >= 9): 
            GY_Cal.append(float(Col[5]))
            GY_Exp.append(float(Col[6]))
            error_cal = abs(float(Col[5])-float(Col[6]))/abs(float(Col[8]))
            GY_Err.append(error_cal)      
            if(abs(float(Col[8])) >= 3.0):
                highlight_indices.append(num-2)
itr = 0
for idx in highlight_indices:
    itr = itr + 1
    plt.annotate(
        text=f"{GY_Label[idx]}",  # Label with the x-value
        xy=(GY_Label[idx], GY_Exp[idx]),  # Point at the data
        xytext=(GY_Label[idx],min(GY_Cal)*itr**10),  # Position of the label (adjust as needed)
        arrowprops=dict(arrowstyle="->", color="white"),
        ha='center'  # Center-align the text
    )

plt.errorbar(GY_Label, GY_Exp,yerr = GY_Err,label= "Experimental", color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );

plt.scatter(GY_Label, GY_Cal, label= "Calulated", color= "cyan",marker= "o", s=30)
# x-axis label
plt.xlabel('Energy (keV)')
plt.yscale("log")   
# frequency label
plt.ylabel('(Gamma Yields)')
# plot title
plt.title('GOSIA: Gamma Yields (CALCULATED VS. EXPERIMENTAL)')
# showing legend
plt.legend()


#------[Mixing Ratio Graphs]--------|

plt.subplot(2, 2, 4)

MIX_Label= []
MIX_Exp = []
MIX_Err = []
MIX_Cal = []
num = 0
for Col in cleaned_data["E2/M1 MIXING RATIOS"]:
    num = num + 1
    if num > 1:
        MIX_Label.append(Col[0]+Col[1])
        MIX_Cal.append(float(Col[3]))
        MIX_Exp.append(float(Col[2]))
        error_cal = abs(float(Col[3])-float(Col[2]))/abs(float(Col[4]))
        MIX_Err.append(error_cal)

plt.errorbar(MIX_Label, MIX_Exp,label= "Experimental",yerr = MIX_Err, color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );
plt.scatter(MIX_Label, MIX_Cal, label= "Calulated", color= "cyan",marker= "o", s=30)
# x-axis label
plt.xlabel('Li-->Lf')
#plt.yscale("log")   
# frequency label
plt.ylabel('(E2/M1) Mixing Ratio')
# plot title
plt.title('GOSIA: E2/M1 Mixing Ratios (CALCULATED VS. EXPERIMENTAL)')
# showing legend
plt.legend()


#---------[Matrix Element Comp]
plt.figure()

ME_Label= []
ME_Exp = []
ME_Err = []
ME_Cal = []
highlight_indices = []
num = 0
for Col in cleaned_data["ME_Cal"]:
    num = num + 1
    if num > 1:
        ME_Label.append(Col[0] +"-"+ Col[1])
        ME_Cal.append(float(Col[3]))
        ME_Exp.append(float(Col[2]))
        ME_Err.append(abs(float(Col[3])-float(Col[2]))/abs(float(Col[4])))
         
        if(abs(float(Col[4])) >= 3.0):
            highlight_indices.append(num-2)
itr = 0
for idx in highlight_indices:
    itr = itr + 1
    plt.annotate(
        text=f"{ME_Label[idx]}",  # Label with the x-value
        xy=(ME_Label[idx], ME_Cal[idx]),  # Point at the data
        xytext=(ME_Label[idx],ME_Cal[idx]-0.5),  # Position of the label (adjust as needed)
        arrowprops=dict(arrowstyle="->", color="white"),
        ha='center'  # Center-align the text
    )

plt.errorbar(ME_Label, ME_Exp,label= "Experimental",yerr = ME_Err, color='red', fmt='o',  # Data points as circles
    ecolor='gray',  # Color of error bars
    capsize=3,  # Size of error bar caps
    elinewidth=2  # Line width of error bars
    );

plt.scatter(ME_Label, ME_Cal, label= "Calulated", color= "cyan",marker= "o", s=30)
# x-axis label
plt.xlabel('Energy (keV)')   
# frequency label
plt.ylabel('Matrix Elements (e2bl)')
# plot title
plt.title('GOSIA: Given Matrix Elements (CALCULATED VS. EXPERIMENTAL)')
# showing legend
plt.legend()
plt.show()

#--------[ Section: Plotting Diagrams/Graphs ]--------------------------||