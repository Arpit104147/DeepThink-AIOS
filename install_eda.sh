#!/bin/bash
echo "Installing EDA Chip Design Tools (Icarus Verilog, Yosys, NGSPICE)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y && sudo apt-get install -y iverilog yosys ngspice klayout
elif command -v brew &> /dev/null; then
    brew install icarus-verilog yosys ngspice
else
    echo "⚠️ Package manager not recognized. Please install iverilog, yosys, ngspice manually."
fi
echo "✅ EDA Toolchain setup complete!"
