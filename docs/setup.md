# Setup Guide

This guide will walk you through the process of setting up your environment to run the code and examples from the book.

## System Requirements

- **Python 3.8+**
- **pip** (Python package installer)
- **git** (Version control system)

## Step 1: Clone the Repository

First, you need to clone the repository to your local machine. Open your terminal and run the following command:

```bash
git clone https://github.com/[your-username]/soccer-analytics-book.git
cd soccer-analytics-book
```

## Step 2: Create a Virtual Environment

It is highly recommended to use a virtual environment to manage the dependencies for this project. This will prevent conflicts with other Python projects on your system.

### Using `venv`

If you are using Python 3, you can create a virtual environment using the built-in `venv` module:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Using `conda`

If you are using Anaconda or Miniconda, you can create a new environment with the following command:

```bash
conda create --name soccer-analytics python=3.9
conda activate soccer-analytics
```

## Step 3: Install Dependencies

Once you have activated your virtual environment, you can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

This will install all of the libraries listed in the `requirements.txt` file, including pandas, NumPy, scikit-learn, and Jupyter.

## Step 4: Verify Your Installation

To make sure everything is set up correctly, you can run the following command to start the Jupyter Notebook server:

```bash
jupyter notebook
```

This should open a new tab in your web browser. From there, you can navigate to the `notebooks/` directory and open one of the notebooks to make sure it runs without errors.

## Step 5: Running Your First Notebook

1. Navigate to the `notebooks/chapter-02/` directory.
2. Open the `01-python-basics.ipynb` notebook.
3. Run the cells in the notebook by selecting them and pressing `Shift + Enter`.

If the code runs without any errors, you are all set up and ready to go!

## Troubleshooting

If you encounter any issues during the setup process, please check the following:

- Make sure you have a compatible version of Python installed.
- Ensure that you have activated your virtual environment before installing the dependencies.
- If you are having trouble with a specific package, try installing it individually (`pip install <package-name>`).

If you are still having trouble, please open an issue in the repository, and we will do our best to help you.
