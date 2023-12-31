# Planning

## Path planning algorithms

Modified from: [https://github.com/zhm-real/PathPlanning.git](https://github.com/zhm-real/PathPlanning.git)

### Credits
The code in this repository is based on the original implementation available at [https://github.com/zhm-real/PathPlanning.git](https://github.com/zhm-real/PathPlanning.git). It has been modified and enhanced for ease of use.

### Getting Started
To use the path planning algorithms in this repository, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/universea/Planning.git
    ```

2. Install the required dependencies:
    ```bash
    python setup.py install
    ```

3. Run the examples:

    For 2D planning, you need to provide a map where pixel values represent the drivable areas and obstacles. The drivable areas should have a pixel value of 0, while obstacles should have pixel values greater than or equal to 1.

    ```bash
    python examples.py
    ```
