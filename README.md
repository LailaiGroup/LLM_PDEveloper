# LLM-PDEveloper [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20465088.svg)](https://doi.org/10.5281/zenodo.20465088)
Wu Haoyang, Ni Wanglai, Zhang Xinxin
## Installation
```
conda env create -n pdeveloper -f environment.yml
```

## Examples
After finish the installation. We can reproduce the experiments through the following steps:
1. Setup the OpenAI API:
```bash
export OPENAI_API=[your/OpenAI/API]
```
2. Activate the environment
```bash
conda activate pdeveloper
```
3. Run the script:
```bash
cd examples/AD/minipack
bash o1-test.sh
```



