# MLP from Scratch

A **clean and educational implementation of a Multi-Layer Perceptron (MLP) neural network from scratch**.

This project aims to provide a deep understanding of the fundamental building blocks of a neural network, including the core concepts of forward propagationand backpropagation without relying on any frameworks.

---

##  Features

* **From Scratch Implementation:** Build a functional MLP using only low-level tools, which makes the model lightweight and fast
* **Hybrid Performance:** Utilizes **Python** for the main logic and user interface and the performance critical components are implemented in `C` for speed.
* **Modular Design:** Clearly separated directories for source code, C extensions, and tests.

---

## Tech Stack

* `Python` The primary language for the MLP implementation.
* `C` Used for optimized, low-level computations (likely via Python bindings).
* `Makefile` For compiling the C source files and managing the build process.
* `Nix` Used for environment management

---

## Local Setup

To get this project running locally, you will need the nix package manager.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:toms55/MLP_from_scratch.git
   ```
2. **Install the packages**
    ```bash
    nix develop
    ```

3.  **Build the C extensions:**
    Run the `make all` command to compile the C source files into Python-loadable modules.
    ```bash
    make all
    ```

4.  **Run the project/tests:**
    Once compiled, you can run any included example scripts or verification tests found in the `tests/` directory.

    ```bash
    python3 python/run.py
    ```
