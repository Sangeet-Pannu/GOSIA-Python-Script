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
     The user must also define the transitions in the level scheme they want updated by the code with the calculated B(E2) values.
     
  2. User must alter some lines in the code:
       1. **line 171: file path and name of .out file**
       2. **line 301: File path of .ags file.**
    

# run_GOSIA2RAD.sh
This bash script will run the previous python code and the gls code, doing the conversion of ASCII to gls, and will display the gls level scheme with updated B(E2) values.

THIS CODE IS TO BE RUN! You can run the GOSIAOUTEX.py but this one just converts the ascii file from AGS to GLS so user does not have to.
