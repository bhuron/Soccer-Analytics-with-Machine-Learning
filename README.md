# Companion Code for Soccer Analytics with Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official companion code for the book **Soccer Analytics with Machine Learning** by **Haipeng Gao, Ari Joury, Guanyu Hu, and Weining Shen**. It provides all the Jupyter Notebooks and supplementary materials needed to follow along with the examples and exercises in the book.

## About the Book

**Soccer Analytics with Machine Learning** is a practical guide to soccer analytics, designed for both data scientists who want to apply their skills to the beautiful game and soccer enthusiasts who want to learn about data analysis. The book covers everything from the fundamentals of Python and data wrangling to advanced machine learning models for predicting match outcomes and evaluating player performance.

**[Link to Publisher/Purchase Page]**

## Quick Start

To get started with the code, you will need to have Python 3.8+ installed on your system. You can then clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/[your-organization]/[your-repo-name].git
cd [your-repo-name]
pip install -r requirements.txt
```

For a more detailed setup guide, please see the **[Setup Documentation](docs/setup.md)**.

## Repository Structure

This repository is organized to mirror the structure of the book, making it easy to find the code for each chapter.

```
soccer-analytics-book/
├── README.md                          # Main landing page with overview
├── LICENSE                            # License information
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── notebooks/                         # Main content directory
│   ├── chapter-02/                    # Python Basics
│   ├── chapter-03/                    # Data Wrangling
│   └── ...                            # And so on for each chapter
│
├── extras/                            # Additional resources
│   ├── extended-examples/             # Beyond-the-book examples
│   └── solutions/                     # Exercise solutions
│
└── docs/                              # Documentation
    └── setup.md                       # Installation guide
```

## Data Sources

This book and its companion notebooks rely primarily on open-source data that can be accessed programmatically. You do not need to download any data files to get started.

### StatsBomb Open Data

The vast majority of examples use data from **StatsBomb's open data initiative**. This is a rich source of event-level data for thousands of matches from various competitions.

- **Source**: [StatsBomb Open Data on GitHub](https://github.com/statsbomb/open-data)
- **License**: Please review the [StatsBomb Open Data License](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) before using the data.

We use the `statsbombpy` Python library to easily access this data directly within the notebooks. The library handles the downloading and parsing of the data for you.

```python
# Example of loading data with statsbombpy
from statsbombpy import sb

# Get all competitions
competitions = sb.competitions()

# Get matches for a specific competition
matches = sb.matches(competition_id=72, season_id=30)
```

### Other Datasets

Some chapters may use other datasets for specific examples (e.g., `shots_data_large.csv`). In these cases, the notebook will either provide a direct download link or include the code necessary to generate the dataset from the StatsBomb data.

## How to Use the Notebooks

Each chapter in the book has a corresponding folder in the `notebooks/` directory. To run the notebooks, you will need to have Jupyter installed. You can start the Jupyter Notebook server by running the following command in your terminal:

```bash
jupyter notebook
```

This will open a new tab in your web browser, where you can navigate to the `notebooks/` directory and open the notebook you want to run.

## Citation

If you use the code from this repository in your own work, please cite the book as follows:

**Haipeng Gao, Ari Joury, et al. Soccer Analytics with Machine Learning. O'Reilly Media, 2026.**

## Contact and Support

If you have any questions, find a bug, or have a suggestion for improvement, please open an issue in this repository. We welcome contributions from the community!
