# Cross-Align
Code for [EMNLP2022 "Cross-Align: Modeling Deep Cross-lingual Interactions for Word
Alignment"](https://aclanthology.org/2022.emnlp-main.244.pdf)

```Cross-Align``` is a high-quality word alignment tool which fully considers 
the cross-lingual context by modeling deep interactions between the input sentence pairs.

The following table shows how it compares to popular alignment models, the best scores are in **bold**:

|                                                   | De-En | En-Fr | Ro-En | Zh-En | Ja-En |
|:---------------------------------------------------------|------:|:-----:|------:|:-----:|:-----:|
| [FastAlign](https://github.com/clab/fast_align)          |  26.2 | 10.5  |  31.4 | 23.7  | 51.1  |
| [GIZA++](http://www2.statmt.org/moses/giza/GIZA++.html)  |  18.9 |  5.5  |  26.6 | 19.4  | 48.0  |
| [SimAlign](https://github.com/cisnlp/simalign)           |  18.8 |  7.6  |  27.2 | 21.6  | 46.6  |
| [Awesome-Align](https://github.com/neulab/awesome-align) |  15.6 |  4.4  |  23.0 | 12.9  | 38.4  | 
| **Ours**                                                 |  **13.6** |  **3.4**  |  **20.9** | **10.1**  | **35.4**  |

We released the above five langauge pairs of Cross-Align models, you can download [HERE](https://drive.google.com/file/d/1FNB37uTLQRr0nXyJ4DdAB01pV_SInzlB/view?usp=sharing) and inference on test data directly.
## Requirements
```
pip install --user --editable ./
```
## Input format
Inputs should be **tokenized** and each line is a source language sentence and 
its target language translation, separated by (```|||```). For example:
```
Das stimmt nicht ! ||| But this is not what happens .
```
## Two-stage Training
Training Cross-Align on parallel data to get good alignments.
### First training stage
In the first stage, the model is trained with TLM to learn the cross-lingual representations.
```
sh ./srcipt/train_stage1.sh
```
### Second training stage
After the first training stage, the model is then finetuned with a self-supervised alignment
objective to bridge the gap between the training and inference.
```
sh ./srcipt/train_stage2.sh
```
## Inference
Extracting word alignments from ```Cross-Align```.
```commandline
sh ./srcipt/inference.sh
```
```Cross-Align``` produces outputs in the widely-used i-j “Pharaoh format,” where a pair i-j indicates that the i-th word (zero-indexed) of 
the source language is aligned to the j-th word of the target sentence. You can see some examples in the ```data/xx.out```.
## Calculating AER
The gold alignment file should have the same format as Cross-Align outputs. For sample parallel sentences and their gold alignments, see ```data/test.xx-xx``` and ```data/xx.talp```.
```commandline
sh ./srcipt/cal_aer.sh
```
## Publication
If you use the code, please cite
```
@inproceedings{lai-etal-2022-cross,
    title = "Cross-Align: Modeling Deep Cross-lingual Interactions for Word Alignment",
    author = "Lai, Siyu  and
      Yang, Zhen  and
      Meng, Fandong  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.244",
    pages = "3715--3725",
}
```
