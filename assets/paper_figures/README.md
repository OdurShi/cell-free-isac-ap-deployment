# Figures of the paper

Cropped from the published PDF with `scripts/extract_paper_figures.py` (300 dpi, figure
without caption).

| File | Figure | Caption as printed in the paper |
|---|---|---|
| `fig01.png` | Figure 1 | Cell-free ISAC systems. |
| `fig02.png` | Figure 2 | Architecture of SAC-based cell-free ISAC APs deployment system. |
| `fig03.png` | Figure 3 | Accumulative ISAC rewards of different DRL Algorithms. (a) Max-sum problem. (b) Max-min problem. |
| `fig04.png` | Figure 4 | APs deployment results of max-sum problem. |
| `fig05.png` | Figure 5 | APs deployment results of max-min problem. |

Figures reproduced from the published article; copyright remains with the publisher/authors
as stated there: Fanfei Xu, Shengheng Liu, Zihuan Mao, Shangqing Shi, Dazhuan Xu, Dongming
Wang, and Yongming Huang. 2024. Access Point Deployment for Localizing Accuracy and User Rate
in Cell-Free Systems. In The 30th Annual International Conference on Mobile Computing and
Networking (ACM MobiCom '24), November 18-22, 2024, Washington D.C., DC, USA. ACM, New York,
NY, USA, 7 pages. https://doi.org/10.1145/3636534.3698221

To regenerate the images:

```bash
pip install pymupdf
python scripts/extract_paper_figures.py /path/to/paper.pdf --out assets/paper_figures
```
