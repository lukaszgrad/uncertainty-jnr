<div align="center">

# _Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer_

**Łukasz Grad**  
University of Warsaw, Poland  
ReSpo.Vision, Poland  
📧 Primary Contact: l.grad@mimuw.edu.pl

[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf)
[![Supplement](https://img.shields.io/badge/Supplement-PDF-red)](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/supplemental/Grad_Single-Stage_Uncertainty-Aware_Jersey_CVPRW_2025_supplemental.pdf)
[![License](https://img.shields.io/badge/License-CC--BY--SA--4.0-green)](LICENSE)

</div>

## 🔍 Overview

**TL; DR.** We present a single-stage uncertainty-aware approach for jersey number recognition in soccer that employs digit-compositional classifiers and Dirichlet-based uncertainty modeling, achieving 85.62% tracklet-level accuracy on the SoccerNet Challenge benchmark.

![Method Overview](https://lukaszgrad.github.io/jnr/static/images/cvsports_method2.png)

## 🚀 Installation

### Requirements
- Python 3.11
- Poetry

### Setup
```bash
# Clone the repository
git clone https://github.com/lukaszgrad/uncertainty-jnr.git
cd uncertainty-jnr

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## 📚 Documentation

- 📊 **[Dataset Information](docs/DATA.md)** - Information about datasets, access instructions, and data structure
- 🏋️ **Training & Evaluation Guides** - *Coming soon*
- 💾 **Model checkpoints** - *Coming soon*

## 🏆 Results

### Comparison with Prior Work on SoccerNet Benchmarks

| Method | Test Acc | Challenge Acc |
|--------|----------|---------------|
| Gerke et al. (2015) | 32.57% | 35.79% |
| Vats et al. (2021) | 46.73% | 49.88% |
| Li et al. (2018) | 47.85% | 50.60% |
| Vats et al. (2022) | 52.91% | 58.45% |
| Balaji et al. (2023) | 68.53% | 73.77% |
| Koshkina et al. (2024) | 87.45% | 79.31% |
| **Ours (ViT-S, SoccerNet Dataset)** | 82.74% | - |
| **Ours (ViT-B, SoccerNet Dataset)** | 86.37% | 83.52% |
| **Ours (ViT-B, 200M Dataset)** | 85.46% | **85.62%** |
| **Ours (ViT-B, 200M + SoccerNet finetuned)** | **88.27%** | 85.41% |

*Tracklet-level accuracy results. Prior work results are taken from Koshkina et al. (2024).*

## ✏️ Citation

If you find this work helpful, please cite our paper:

```bibtex
@InProceedings{Grad_2025_CVPR,
    author    = {Grad, {\L}ukasz},
    title     = {Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {6102-6110}
}
```

## 📄 License

This project is licensed under the [CC-BY-SA-4.0](LICENSE) license.