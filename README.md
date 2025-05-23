Created by Sangeet-Pal Pannu

# GOSIA VISUALIZATION CODE


The codes are implemented to help user's visualize the .out file when running minimization of GOSIA with experimental information such as Yields, Branching Ratios, Lifetimes, Mixing ratios, and Known Matrix Elements.


# GOSIAOUTEX.py usage instructions

### USER MUST HAVE THE FOLLOWING PYTHON LIBRARIES:
usually when installing python3 specific libraries such as numpy and math are included.

1. mplcursors
2. pandas


This code has two functionalities:
  1. Read the .out file and produces pyplots comparing experimental values to Calculated values from GOSIA.
  2. Read .ags file (ASCII RADWARE FILE) and produces a gls level scheme in which the transitions are labelled with B(E2) values.


### Things to Note when running the Code:

  1. User must create a gls file level scheme with RADWARE, in which the Level energies coincide directly with GOSIA level energies.
     The user must also define the transitions in the level scheme they want updated by the code with the calculated B(E2) values. In the end, once the user
     has defined the level scheme, will have to use gls_conv to convert GLS to ASCII.
     
  3. User must alter some lines in the code:
       1. **line 171: file path and name of .out file**
       2. **line 301: File path of .ags file.**
    

# run_GOSIA2RAD.sh
This bash script will run the previous python code and the gls code, doing the conversion of ASCII to gls, and will display the gls level scheme with updated B(E2) values.

THIS CODE IS TO BE RUN! You can run the GOSIAOUTEX.py but this one just converts the ascii file from AGS to GLS so user does not have to.


# OUTPUT OF CODE FOR GOSIA RUN OF 110CD
NOTE: The run shown here is a random run with no significance to the ongoing analysis.
The also has the ability to hover over lifetime data points to reveal the lifetime value and is able to point (with a white arrow) towards
data points where the difference between the experiment and calculated is > 3*Sigma.

EXPERIMENTAL VS. CALCULATED PLOTS:
![image](https://github.com/user-attachments/assets/b6131258-271f-4e1a-a469-151432d2468f)

MATRIX ELEMENTS (KNOWN):

![image](https://github.com/user-attachments/assets/cf2b16fb-2fb9-40e0-8e1c-7a06facc5420)

GLS FILE:
![image](https://github.com/user-attachments/assets/35232aa8-9169-4bb1-bc6c-a9e348b82b98)

