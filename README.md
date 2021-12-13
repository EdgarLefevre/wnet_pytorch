# Wnet_pytorch

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) 
[![License](https://img.shields.io/github/license/EdgarLefevre/Wnet_pytorch?label=license)](https://github.com/EdgarLefevre/Wnet_pytorch/blob/main/LICENSE)
<a href="https://gitmoji.dev">
  <img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square" alt="Gitmoji">
</a>
<!-- [![PyPI](https://img.shields.io/pypi/v/napari-deepmeta.svg?color=green)](https://pypi.org/project/napari-deepmeta)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-deepmeta.svg?color=green)](https://python.org)
[![tests](https://github.com/EdgarLefevre/napari-deepmeta/workflows/tests/badge.svg)](https://github.com/EdgarLefevre/napari-deepmeta/actions)
[![codecov](https://codecov.io/gh/EdgarLefevre/napari-deepmeta/branch/main/graph/badge.svg?token=H41ZaCAg31)](https://codecov.io/gh/EdgarLefevre/napari-deepmeta)
-->

## Installation 

```bash
conda env create -f environment.yml
```

## usage

```bash
python -m wnet.train # train_v3 is the best atm
```

## Folder explaination

```bash
.
├── data  #output folder
├── docs  #documentation folder (useless atm)
├── test  #test folder (useless atm)
└── wnet  #src folder
    ├── models
    └── utils 
```