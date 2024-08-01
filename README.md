<div align="center">
  <img src="imgs/AIGTD-logo.png" alt="AIGTD Survey" width="500"><br>
  A Comprehensive Survey on Recent Advances in AI-Generated Text Detection
</div>

# Collection of papers and resources for AIGTD
The papers are organized according to our survey: AIGTD: A Comprehensive Survey on Recent Advances in AI-Generated Text Detection. 

The classification topology of AIGTD is constructed based on addressing three key challenges: classifier training, inherent attributes, and information embedding.

**Note:** We will keep updating to make this survey perfect. 

## Table of Contents
- [Tackling Classifier Training](#tackling-classifier-training)
- [Tackling Intrinsic Attributes](#tackling-intrinsic-attributes)
- [Tackling Information Embedding](#tackling-information-embedding)
- [Dataset collation](#dataset-collation)
- [Citation](#citation)

## Tackling Classifier Training
1. Chuck Rosenberg, Martial Hebert, and Henry Schneiderman. Semisupervised self-training of object detection models. 2005. [[paper] (https://www.ri.cmu.edu/pub_files/pub4/rosenberg_charles_2005_1/rosenberg_charles_2005_1.pdf)]

2. Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769, 2016. [[paper] (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)]

3. Beliz Gunel, Jingfei Du, Alexis Conneau, and Ves Stoyanov. Supervised contrastive learning for pre-trained language model fine-tuning. arXiv preprint arXiv:2011.01403, 2020. [[paper](https://arxiv.org/pdf/2011.01403)]

### Feature Analysis

#### Structural-based Analysis
1. Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. Coco: Coherence-enhanced machine-generated text detection under data limitation with contrastive learning. arXiv preprint arXiv:2212.10341, 2022. [[paper](https://arxiv.org/pdf/2212.10341)]

#### Partial Access
1. Yi Xu, Jie Hu, Zhiqiao Gao, and Jinpeng Chen. Ucl-ast: Active self-training with uncertainty-aware clouded logits for few-shot text classification. In 2022 IEEE 34th International Conference on Tools with Artificial Intelligence (ICTAI), pages 1390–1395. IEEE, 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10097942)]

2. Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. Seqxgpt: Sentence-level ai-generated text detection. arXiv preprint arXiv:2310.08903, 2023. [[paper](https://arxiv.org/pdf/2310.08903)]

#### Network Reconstruction
1. Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, and Zhouwang Yang. Are ai-generated text detectors robust to adversarial perturbations? arXiv preprint arXiv:2406.01179, 2024. [[paper](https://arxiv.org/pdf/2406.01179)]

### Probability and Statistics

#### Probability-based Model
1. Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Llmdet: A large language models detection tool. arXiv preprint arXiv:2305.15004, 2023. [[paper](https://arxiv.org/pdf/2305.15004)]

2. Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text ghostwritten by large language models. arXiv preprint arXiv:2305.15047, 2023. [[paper](https://arxiv.org/pdf/2305.15047)]

### Deep Learning

#### Positiva Unlabeled
1. Yuchuan Tian, Hanting Chen, Xutao Wang, Zheyuan Bai, Qinghua Zhang, Ruifeng Li, Chao Xu, and Yunhe Wang. Multiscale positive-unlabeled detection of ai-generated texts. arXiv preprint arXiv:2305.18149, 2023. [[paper](https://arxiv.org/pdf/2305.18149)]

#### Adversarial Training
1. Ying Zhou, Ben He, and Le Sun. Humanizing machine-generated content: Evading ai-text detection through adversarial attack. arXiv preprint arXiv:2404.01907, 2024. [[paper](https://arxiv.org/pdf/2404.01907)]

2. Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Radar: Robust ai-text detection via adversarial learning. Advances in Neural Information Processing Systems, 36:15077–15095, 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/30e15e5941ae0cdab7ef58cc8d59a4ca-Paper-Conference.pdf)]

#### Transfer Training
1. Eric Chu, Jacob Andreas, Stephen Ansolabehere, and Deb Roy. Language models trained on media diets can predict public opinion. arXiv preprint arXiv:2303.16779, 2023. [[paper](https://arxiv.org/pdf/2303.16779)]

2. Hans WA Hanley and Zakir Durumeric. Machine-made media: Monitoring the mobilization of machine-generated articles on misinformation and mainstream news websites. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 542–556, 2024. [[paper](https://ojs.aaai.org/index.php/ICWSM/article/view/31333/33493)]

3. Amrita Bhattacharjee, Tharindu Kumarage, Raha Moraffah, and Huan Liu. Conda: Contrastive domain adaptation for ai-generated text detection. arXiv preprint arXiv:2309.03992, 2023. [[paper](https://arxiv.org/pdf/2309.03992)]

#### BERT-based
1. Hao Wang, Jianwei Li, and Zhengyu Li. Ai-generated text detection and classification based on bert deep learning algorithm. arXiv preprint arXiv:2405.16422, 2024. [[paper](https://arxiv.org/pdf/2405.16422)]



|       Name       | Black box | White box | Unknown |
| :--------------: | :-------: | :-------: | :-----: |
|       GCN        |     ✔️     |           |         |
| Logits as waves  |           |     ✔️     |         |
|     SeqXGPT      |           |     ✔️     |         |
|       SCRN       |           |     ✔️     |         |
| Proxy perplexity |     ✔️     |           |         |
|   Ghostbuster    |     ✔️     |           |         |
|       MPU        |           |     ✔️     |         |
|      RADAR       |     ✔️     |           |         |
|      conDA       |     ✔️     |           |         |
|       BERT       |           |     ✔️     |         |


## Tackling Intrinsic Attributes


### Feature Extraction
1. Farhad Pourpanah, Moloud Abdar, Yuxuan Luo, Xinlei Zhou, Ran Wang, Chee Peng Lim, Xi-Zhao Wang, and QM Jonathan Wu. A review of generalized zero-shot learning methods. IEEE transactions on pattern analysis and machine intelligence, 45(4):4051–4070, 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9832795)]
2. Wei Wang, Vincent W Zheng, Han Yu, and Chunyan Miao. A survey of zero-shot learning: Settings, methods, and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 10(2):1–37, 2019. [[paper](https://dl.acm.org/doi/pdf/10.1145/3293318?casa_token=C2rZx_nhOOwAAAAA:FgIsDW_L0FRdQhZxC59XvJfp9S4P8AXZqW00NHz7gEW8JeWj7sqnUAFTOmYvuwyx_vlnhVhe4swv)]
3. Jinyan Su, Terry Yue Zhuo, Di Wang, and Preslav Nakov. Detectllm: Leveraging log rank information for zero-shot detection of machinegenerated text. arXiv preprint arXiv:2306.05540, 2023. [[paper](https://arxiv.org/pdf/2306.05540)]
4. Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, and Haifeng Chen. Dna-gpt: Divergent n-gram analysis for training-free detection of gpt-generated text. arXiv preprint arXiv:2305.17359, 2023. [[paper](https://arxiv.org/pdf/2305.17359)]
5. Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, and Irina Piontkovskaya. Intrinsic dimension estimation for robust detection of ai-generated texts. Advances in Neural Information Processing Systems, 36, 2024. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7baa48bc166aa2013d78cbdc15010530-Paper-Conference.pdf)]

### Probability-based


|      Name       | Black box | White box | Unknown |
| :-------------: | :-------: | :-------: | :-----: |
|       LRR       |           |     ✔️     |         |
|     N-Gram      |     ✔️     |           |         |
| Inter Dimension |     ✔️     |           |         |
|    DetectGPT    |     ✔️     |           |         |
|    OPT-125M     |           |           |    ✔️    |
|   Divergence    |           |     ✔️     |         |
|    Curvature    |     ✔️     |     ✔️     |         |
|       MMD       |     ✔️     |           |         |
|      BERT       |     ✔️     |           |         |
|     ChatGPT     |     ✔️     |           |         |
|     Mixing      |           |     ✔️     |         |



## Tackling Information Embedding
1. Mercan Topkara, Cuneyt M Taskiran, and Edward J Delp III. Natural language watermarking. In Security, Steganography, and Watermarking of Multimedia Contents VII, volume 5681, pages 441–452. SPIE, 2005. [[paper](https://www.cerias.purdue.edu/tools_and_resources/bibtex_archive/archive/PSI000441.pdf)]

2. Umut Topkara, Mercan Topkara, and Mikhail J Atallah. The hiding virtues of ambiguity: quantifiably resilient watermarking of natural language text through synonym substitutions. In Proceedings of the 8th workshop on Multimedia and security, pages 164–174, 2006. [[paper](http://umut.topkara.org/papers/ToToAt_MMSEC06.pdf)]

3. Xi Yang, Jie Zhang, Kejiang Chen, Weiming Zhang, Zehua Ma, Feng Wang, and Nenghai Yu. Tracing text provenance via context-aware lexical substitution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 11613–11621, 2022. [[paper](https://cdn.aaai.org/ojs/21415/21415-13-25428-1-2-20220628.pdf)]

4. Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu. Watermarking text generated by black-box language models. arXiv preprint arXiv:2305.08883, 2023. [[paper](https://arxiv.org/pdf/2305.08883)]

5. Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, and Jiaheng Zhang. Provably robust multi-bit watermarking for ai-generated text via error correction code. arXiv preprint arXiv:2401.16820, 2024. [[paper](https://arxiv.org/pdf/2401.16820)]

### Training-free

#### Logits Deviation
1. John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pages 17061–17084. PMLR, 2023. [[paper](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)]

2. Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. arXiv preprint arXiv:2306.17439, 2023. [[paper](https://arxiv.org/pdf/2306.17439)]

#### Hash-based
1. Abe Bohan Hou, Jingyu Zhang, Tianxing He, Yichen Wang, YungSung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. arXiv preprint arXiv:2310.03991, 2023. [[paper](https://arxiv.org/pdf/2310.03991)]

2. Yihan Wu, Zhengmian Hu, Hongyang Zhang, and Heng Huang. Dipmark: A stealthy, efficient and resilient watermark for large language models. arXiv preprint arXiv:2310.07710, 2023. [[paper](https://arxiv.org/pdf/2310.07710)]

3. Abe Bohan Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, and Tianxing He. k-semstamp: A clustering-based semantic watermark for detection of machine-generated text. arXiv preprint arXiv:2402.11399, 2024. [[paper](https://arxiv.org/pdf/2402.11399)]

#### Message Decoding
1. Xuandong Zhao, Lei Li, and Yu-Xiang Wang. Permute-and-flip: An optimally robust and watermarkable decoder for llms. arXiv preprint arXiv:2402.05864, 2024. [[paper](https://arxiv.org/pdf/2402.05864)]

2. Scott Aaronson, Jiahui Liu, Qipeng Liu, Mark Zhandry, and Ruizhe Zhang. New approaches for quantum copy-protection. In Advances in Cryptology–CRYPTO 2021: 41st Annual International Cryptology Conference, CRYPTO 2021, Virtual Event, August 16–20, 2021, Proceedings, Part I 41, pages 526–555. Springer, 2021. [[paper](https://arxiv.org/pdf/2004.09674)]

###  Training-based
#### Message Encoding
1. Han Fang, Zhaoyang Jia, Hang Zhou, Zehua Ma, and Weiming Zhang. Encoded feature enhancement in watermarking network for distortion in real scenes. IEEE Transactions on Multimedia, 2022. [[paper](http://staff.ustc.edu.cn/~zhangwm/Paper/2022_19.pdf)]

2. Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, and Farinaz Koushanfar. Remark-llm: A robust and efficient watermarking framework for generative large language models. arXiv preprint arXiv:2310.12362, 2023. [[paper](https://arxiv.org/pdf/2310.12362)]

### Information Capacity

#### Multi-bit
1. KiYoon Yoo, Wonhyuk Ahn, Jiho Jang, and Nojun Kwak. Robust multi-bit natural language watermarking through invariant features. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2092–2115, 2023. [[paper](https://arxiv.org/pdf/2305.01904)]

2. Pierre Fernandez, Antoine Chaffin, Karim Tit, Vivien Chappelier, and Teddy Furon. Three bricks to consolidate watermarks for large language models. In 2023 IEEE International Workshop on Information Forensics and Security (WIFS), pages 1–6. IEEE, 2023. [[paper](https://arxiv.org/pdf/2308.00113)]

3. Massieh Kordi Boroujeny, Ya Jiang, Kai Zeng, and Brian Mark. Multi-bit distortion-free watermarking for large language models. arXiv preprint arXiv:2402.16578, 2024. [[paper](https://arxiv.org/pdf/2402.16578)]

## Dataset collation

|   Datasets   |  Size   |               Data Description               |
| :----------: | :-----: | :------------------------------------------: |
| TuringBench  |   200   |                News articles                 |
|     HC3      | 44,425  |   Reddit, Wikipedia, medicine and finance    |
|    CHEAT     | 35,304  |              Academic abstracts              |
| Ghostbuster  | 12,685  |  Student essays, creative fiction, and news  |
| GPT-Sentinel | 29,395  |                 OpenWebText                  |
|      M4      | 122,481 |                Multi-domains                 |
|   MGTBench   |  2,817  |         Question-answering datasets          |
|   HC3 Plus   | 214,498 | Summarization, translation, and paraphrasing |
|  MULTITuDE   | 74,081  |                 MassiveSumm                  |
|      \-      |  1,378  |        Public dataset, unknown source        |
|      \-      |   \-    |       Self-built hotel review dataset        |
|   OpenGen    |  3,000  | 3,000 randomly selected two-sentence blocks  |
|   C4 News    |   \-    |                15GB news data                |
|    Alpaca    |   \-    |    Used for question and answer task test    |
|      C4      |   \-    |     Used for open text generation tasks      |
|    Grover    |   \-    |     News generator based on Transformer      |
|     XSum     |   \-    |                News articles                 |

## Citation
If you find this project useful in your research or work, please consider citing it:

```
Coming soon...
```


## Acknowledgements
Your contributions will be acknowledged. 

[Github Flavored Markdown](https://github.com/guodongxiaren/README)
