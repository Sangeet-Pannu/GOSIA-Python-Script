#!/bin/bash

touch counter
TMP=./counter
echo "$(($(cat $TMP) + 1))">$TMP

# Run the Python program
python3 GOSIAOUTEX.py &

# Run gls_conv and provide inputs automatically
gls_conv << EOF
4
items.ags
test_$(cat $TMP).gls
y
EOF

gls test_$(cat $TMP).gls

# Notify the user of completion
echo "Python program and gls_conv executed successfully!"

