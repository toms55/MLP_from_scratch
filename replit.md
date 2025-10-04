# Neural Network from Scratch (C + Python)

## Overview
This project is a Feed-forward Neural Network (FNN) implementation with backpropagation built from scratch in C and Python. The C backend provides optimized matrix operations and activation functions, while Python provides the high-level interface for defining and training neural networks.

**Current State:** Working neural network implementation that trains on the California Housing dataset for regression tasks.

## Recent Changes
- **2025-10-04:** Fixed critical memory bugs in C library:
  - Fixed `free_matrix` function to properly free memory
  - Corrected MSE loss function to handle transposed matrices correctly
  - All matrix operations and neural network training now working properly

## Project Architecture

### Core Components

#### 1. C Backend (`c_src/`)
- **Matrix Operations** (`matrix.c`): Core matrix operations (add, subtract, multiply, transpose, hadamard product)
- **Activation Functions** (`activation.c`): Sigmoid, ReLU, and their derivatives
- **Loss Functions** (`loss.c`): Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE)
- **Compiled Output**: Shared library `build/libmlp.so`

#### 2. Python Frontend (`python/`)
- **C Wrapper** (`c_wrapper.py`): ctypes-based interface to C library
- **MLP Class** (`mlp.py`): High-level neural network implementation
- **Training Script** (`run.py`): Example training on California Housing dataset

### How It Works
1. Python defines the network architecture (layer sizes, activations, loss function)
2. Python converts NumPy arrays to C matrices via ctypes
3. C library performs all heavy computations (forward/backward pass, matrix ops)
4. Python orchestrates training loop and data preprocessing
5. Results are converted back to NumPy for analysis/visualization

## Project Structure
```
.
├── c_src/
│   ├── include/           # C header files
│   │   ├── activation.h
│   │   ├── loss.h
│   │   └── matrix.h
│   └── src/              # C source files
│       ├── activation.c
│       ├── loss.c
│       └── matrix.c
├── python/
│   ├── __init__.py
│   ├── c_wrapper.py      # ctypes interface to C
│   ├── mlp.py           # Neural network class
│   └── run.py           # Training example
├── tests/
│   ├── main.c           # C unit tests
│   └── tests.py         # Python tests
├── build/               # Compiled C library (gitignored)
├── Makefile            # Build system
└── flake.nix          # Nix development environment
```

## Setup & Running

### Building the C Library
```bash
make clean
make all
```
This compiles the C source files and creates `build/libmlp.so`.

### Running the Neural Network
The workflow "Neural Network Training" runs:
```bash
python python/run.py
```

This will:
- Load California Housing dataset
- Create a neural network with architecture: [8, 20, 10, 1]
- Train for 300 epochs using ReLU activation and MSE loss
- Display predictions and evaluation metrics
- Generate a scatter plot of true vs predicted values

### Running C Tests
```bash
make test      # Build test executable
make run       # Run tests
```

## Dependencies

### System
- GCC (C compiler)
- Python 3.11

### Python Packages
- numpy
- pandas
- scipy
- matplotlib
- scikit-learn

All dependencies are automatically installed via the Replit environment.

## Technical Details

### Memory Management
- All C matrices are allocated dynamically
- Python wrapper tracks Matrix objects with rows, cols, and c_ptr
- Proper cleanup with `free_py_matrix()` to avoid memory leaks

### Activation Functions
- **ReLU**: Used for hidden layers, outputs max(0, x)
- **Sigmoid**: Alternative activation, outputs 1/(1+exp(-x))
- **Identity**: Linear activation for regression output

### Training
- Uses mini-batch gradient descent
- Xavier/Glorot weight initialization
- Standard feature scaling on inputs
- MSE loss for regression tasks

## Known Issues & Limitations
- Currently only supports regression (MSE loss)
- No support for classification with cross-entropy
- No GPU acceleration
- Single-threaded execution
- Plot display (`plt.show()`) may not work in Replit console environment

## Future Enhancements
- Add cross-entropy loss for classification
- Implement additional optimizers (Adam, RMSprop)
- Add regularization (L1/L2)
- Support for different batch sizes
- Save/load trained models
