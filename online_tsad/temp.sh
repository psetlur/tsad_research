#!/bin/bash

# Run main.py
echo "Running main.py..."
python src/main.py

# Check if main.py ran successfully before moving to main2.py
if [ $? -eq 0 ]; then
    echo "main.py completed successfully, now running main2.py..."
    python src/main2.py

    # Check if main2.py ran successfully before moving to main3.py
    if [ $? -eq 0 ]; then
        echo "main2.py completed successfully, now running main3.py..."
        python src/main3.py
    else
        echo "main2.py failed. Skipping main3.py."
        exit 1
    fi
else
    echo "main.py failed. Skipping main2.py and main3.py."
    exit 1
fi
