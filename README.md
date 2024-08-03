<div align="center">
  <img src="imgs/AIGTD-logo.png" alt="AIGTD Survey" width="500"><br>
  A Comprehensive Survey on Recent Advances in AI-Generated Text Detection
</div>

# Collection of papers and resources for AIGTD
The papers are organized according to our AIGTD survey: [The Imitation Game Revisited: A Comprehensive Survey on Recent Advances in AI-Generated Text Detection](https://github.com/Nicozwy/AIGTD-Survey/edit/main). 

The classification topology of AIGTD is constructed based on addressing three key challenges: classifier training, inherent attributes, and information embedding.

**Note:** We will keep updating to make this survey perfect. 

## Table of Contents
- [Tackling Classifier Training](#tackling-classifier-training)
- [Tackling Intrinsic Attributes](#tackling-intrinsic-attributes)
- [Tackling Information Embedding](#tackling-information-embedding)
- [Dataset collation](#dataset-collation)
- [Citation](#citation)

## Tackling Classifier Training
1. Chuck Rosenberg, Martial Hebert, and Henry Schneiderman. Semisupervised self-training of object detection models. In Proceedings of the Seventh IEEE Workshops on Application of Computer Vision, pages 29–36, 2005. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4129456)]

2. Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 761–769, 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780458)]

3. Beliz Gunel, Jingfei Du, Alexis Conneau, and Veselin Stoyanov. Supervised contrastive learning for pre-trained language model finetuning. In International Conference on Learning Representations, pages 1–15, 2021. [[paper](https://openreview.net/pdf?id=cu7IUiOhujH)]

### Feature Analysis

#### Structural-based Analysis
1. Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. Coco: Coherence-enhanced machine-generated text detection under data limitation with contrastive learning. arXiv preprint arXiv:2212.10341, 2022. [[paper](https://arxiv.org/pdf/2212.10341)]

#### Partial Access
1. Yi Xu, Jie Hu, Zhiqiao Gao, and Jinpeng Chen. Ucl-ast: Active self-training with uncertainty-aware clouded logits for few-shot text classification. In 2022 IEEE 34th International Conference on Tools with Artificial Intelligence (ICTAI), pages 1390–1395. IEEE, 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10097942)]

2. Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. Seqxgpt: Sentence-level ai-generated text detection. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 1144–1156, 2023. [[paper](https://arxiv.org/pdf/2310.08903)]

#### Network Reconstruction
1. Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, and Zhouwang Yang. Are ai-generated text detectors robust to adversarial perturbations? arXiv preprint arXiv:2406.01179, 2024. [[paper](https://arxiv.org/pdf/2406.01179)]

### Probability and Statistics

#### Probability-based Model
1. Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Llmdet: A third party large language models generated text detection tool. In Findings of the Association for Computational Linguistics: EMNLP, pages 2113–2133, 2023. [[paper](https://arxiv.org/pdf/2305.15004)]

2. Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, pages 1702–1717, 2024. [[paper](https://arxiv.org/pdf/2305.15047)]

### Deep Learning

#### Positiva Unlabeled
1. Yuchuan Tian, Hanting Chen, Xutao Wang, Zheyuan Bai, Qinghua Zhang, Ruifeng Li, Chao Xu, and Yunhe Wang. Multiscale positive unlabeled detection of ai-generated texts. In International Conference on Learning Representations, 2024. [[paper](https://openreview.net/pdf?id=5Lp6qU9hzV)]

#### Adversarial Training
1. Ying Zhou, Ben He, and Le Sun. Humanizing machine-generated content: Evading ai-text detection through adversarial attack. In Proceedings of the Joint International Conference on Computational Linguistics, Language Resources and Evaluation, pages 8427–8437, 2024. [[paper](https://arxiv.org/pdf/2404.01907)]

2. Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Radar: Robust ai-text detection via adversarial learning. Advances in Neural Information Processing Systems, 36:15077–15095, 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/30e15e5941ae0cdab7ef58cc8d59a4ca-Paper-Conference.pdf)]

#### Transfer Training
1. Eric Chu, Jacob Andreas, Stephen Ansolabehere, and Deb Roy. Language models trained on media diets can predict public opinion. arXiv preprint arXiv:2303.16779, 2023. [[paper](https://arxiv.org/pdf/2303.16779)]

2. Hans WA Hanley and Zakir Durumeric. Machine-made media: Monitoring the mobilization of machine-generated articles on misinformation and mainstream news websites. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 542–556, 2024. [[paper](https://ojs.aaai.org/index.php/ICWSM/article/view/31333/33493)]

3. Amrita Bhattacharjee, Tharindu Kumarage, Raha Moraffah, and Huan Liu. Conda: Contrastive domain adaptation for ai-generated text detection. In Proceedings of the International Joint Conference on Natural Language Processing and the Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 598610, 2023. [[paper](https://arxiv.org/pdf/2309.03992)]

#### BERT-based
1. Hao Wang, Jianwei Li, and Zhengyu Li. Ai-generated text detection and classification based on bert deep learning algorithm. arXiv preprint arXiv:2405.16422, 2024. [[paper](https://arxiv.org/pdf/2405.16422)]



|                                                                 Name                                                                  | Black box | White box |
|:-------------------------------------------------------------------------------------------------------------------------------------:| :-------: | :-------: |
|                                            GCN [[paper](https://arxiv.org/pdf/2212.10341)]                                            |     ✔️     |           |
|                          Logits as waves [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10097942)]                           |           |     ✔️     |
|                                          SeqXGPT [[paper](https://arxiv.org/pdf/2310.08903)]                                          |           |     ✔️     |
|                                           SCRN [[paper](https://arxiv.org/pdf/2406.01179)]                                            |           |     ✔️     |
|                                     Proxy perplexity [[paper](https://arxiv.org/pdf/2305.15004)]                                      |     ✔️     |           |
|                                        Ghostbuster [[paper](https://arxiv.org/pdf/2305.15047)]                                        |     ✔️     |           |
|                                            MPU [[paper](https://arxiv.org/pdf/2305.18149)]                                            |           |     ✔️     |
| RADAR [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/30e15e5941ae0cdab7ef58cc8d59a4ca-Paper-Conference.pdf)] |     ✔️     |           |
|                                           conDA [[paper](https://arxiv.org/pdf/2309.03992)]                                           |     ✔️     |           |
|                                           BERT [[paper](https://arxiv.org/pdf/2405.16422)]                                            |           |     ✔️     |


## Tackling Intrinsic Attributes
1. Nathan Benaich and Ian Hogarth. State of ai report. London, UK, 2020. [[paper]](https://docs.google.com/presentation/d/1ZUimafgXCBSLsgbacd6-a-dqO7yLyzIl1ZJbiCBUUT4/edit#slide=id.g557254d430_0_0)
2. Yuhong Mo, Hao Qin, Yushan Dong, Ziyi Zhu, and Zhenglin Li. Large language model (llm) ai text generation detection based on transformer deep learning algorithm. International Journal of Engineering and Management Research, 14(2):154–159, 2024. [[paper]](https://arxiv.org/pdf/2405.06652)
3. Rongsheng Wang, Haoming Chen, Ruizhe Zhou, Han Ma, Yaofei Duan, Yanlan Kang, Songhua Yang, Baoyu Fan, and Tao Tan. Llm-detector: Improving ai-generated chinese text detection with open-source llm instruction tuning. arXiv preprint arXiv:2402.01158, 2024. [[paper]](https://arxiv.org/pdf/2402.01158)
4. Farhad Pourpanah, Moloud Abdar, Yuxuan Luo, Xinlei Zhou, Ran Wang, Chee Peng Lim, Xi-Zhao Wang, and QM Jonathan Wu. A review of generalized zero-shot learning methods. IEEE transactions on pattern analysis and machine intelligence, 45(4):4051–4070, 2022. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9832795)
5. Wei Wang, Vincent W Zheng, Han Yu, and Chunyan Miao. A survey of zero-shot learning: Settings, methods, and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 10(2):1–37, 2019. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3293318?casa_token=C2rZx_nhOOwAAAAA:FgIsDW_L0FRdQhZxC59XvJfp9S4P8AXZqW00NHz7gEW8JeWj7sqnUAFTOmYvuwyx_vlnhVhe4swv)


### Feature Extraction

#### Logarithmic Ranking

1. Jinyan Su, Terry Zhuo, Di Wang, and Preslav Nakov. Detectllm: Leveraging log rank information for zero-shot detection of machine generated text. In Findings of the Association for Computational Linguistics: EMNLP, pages 12395–12412, 2023. [[paper]](https://arxiv.org/pdf/2306.05540)

#### N-gram with BScore

1. Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, and Haifeng Chen. Dna-gpt: Divergent n-gram analysis for training-free detection of gpt-generated text. In International Conference on Learning Representations, pages 1–26, 2024. [[paper]](https://openreview.net/pdf?id=Xlayxj2fWp)

#### Internal Dimension
1. Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, and Irina Piontkovskaya. Intrinsic dimension estimation for robust detection of ai-generated texts. Advances in Neural Information Processing Systems, 36, 2024. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/7baa48bc166aa2013d78cbdc15010530-Paper-Conference.pdf)

### Probability-based

#### Conditional Probability

1. Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pages 24950–24962. PMLR, 2023. [[paper]](https://proceedings.mlr.press/v202/mitchell23a/mitchell23a.pdf)
2. Shengchao Liu, Xiaoming Liu, Yichen Wang, Zehua Cheng, Chengzhengxu Li, Zhaohan Zhang, Yu Lan, and Chao Shen. Does∖textsc {DetectGPT} fully utilize perturbation? selective perturbation on model-based contrastive learning detector would be better. arXiv preprint arXiv:2402.00263, 2024. [[paper]](https://arxiv.org/pdf/2402.00263)


#### Probability Curvature

1. Niloofar Mireshghallah, Justus Mattern, Sicun Gao, Reza Shokri, and Taylor Berg-Kirkpatrick. Smaller language models are better black-box machine-generated text detectors. arXiv preprint arXiv:2305.09859, 2023. [[paper]](https://arxiv.org/pdf/2305.09859)
2. Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pages 24950–24962. PMLR, 2023. [[paper]](https://proceedings.mlr.press/v202/mitchell23a/mitchell23a.pdf)
3. Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, and Haifeng Chen. Dna-gpt: Divergent n-gram analysis for training-free detection of gpt-generated text. In International Conference on Learning Representations, pages 1–26, 2024. [[paper]](https://openreview.net/pdf?id=Xlayxj2fWp)
4. Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. Fast-detectgpt: Efficient zero-shot detection of machine generated text via conditional probability curvature. In International Conference on Learning Representations, pages 1–23, 2024. [[paper]](https://openreview.net/pdf?id=Bpcgcr8E8Z)

#### Distribution Difference

1. Shuhai Zhang, Feng Liu, Jiahao Yang, Yifan Yang, Changsheng Li, Bo Han, and Mingkui Tan. Detecting machine-generated texts by multi-population aware optimization for maximum mean discrepancy. arXiv preprint arXiv:2402.16041, 2024. [[paper]](https://arxiv.org/pdf/2402.16041)

### Epidemic Model

#### BERT

1. Utsho Chakraborty, Jaydeep Gheewala, Sheshang Degadwala, Dhairya Vyas, and Mukesh Soni. Safeguarding authenticity in text with bert-powered detection of ai-generated content. In 2024 International Conference on Inventive Computation Technologies (ICICT), pages 34–37. IEEE, 2024 [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10544590)

#### ChatGPT

1. David M Markowitz, Jeffrey T Hancock, and Jeremy N Bailenson. Linguistic markers of inherently false ai communication and intentionally false human communication: Evidence from hotel reviews. Journal of Language and Social Psychology, 43(1):63–82, 2024. [[paper]](https://journals.sagepub.com/doi/full/10.1177/0261927X231200201)

#### Model Mixing

1. Yuhong Mo, Hao Qin, Yushan Dong, Ziyi Zhu, and Zhenglin Li. Large language model (llm) ai text generation detection based on transformer deep learning algorithm. International Journal of Engineering and Management Research, 14(2):154–159, 2024. [[paper]](https://ijemr.vandanapublications.com/index.php/ijemr/article/view/1565/1436)

|                                                                      Name                                                                       | Black box | White box |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:| :-------: | :-------: |
|                                                 LRR [[paper](https://arxiv.org/pdf/2306.05540)]                                                 |           |     ✔️     |
|                                               N-Gram [[paper](https://openreview.net/pdf?id=Xlayxj2fWp)]                                                |     ✔️     |           |
| Inter Dimension [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7baa48bc166aa2013d78cbdc15010530-Paper-Conference.pdf)] |     ✔️     |           |         |
|              DetectGPT [[paper](https://proceedings.mlr.press/v202/mitchell23a/mitchell23a.pdf)] [[paper](https://arxiv.org/pdf/2402.00263)]               |     ✔️     |           |
|                                              OPT-125M [[paper](https://arxiv.org/pdf/2305.09859)]                                               |     ✔️     |           |
|                                             Divergence [[paper](https://openreview.net/pdf?id=Xlayxj2fWp)]                                              |           |     ✔️     |
|                                              Curvature [[paper](https://openreview.net/pdf?id=Bpcgcr8E8Z)]                                              |     ✔️     |     ✔️     |
|                                                 MMD [[paper](https://arxiv.org/pdf/2402.16041)]                                                 |     ✔️     |           |
|                                     BERT [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10544590)]                                      |     ✔️     |           |
|                                ChatGPT [[paper](https://journals.sagepub.com/doi/pdf/10.1177/0261927X231200201)]                                |     ✔️     |           |
|                                               Mixing [[paper](https://ijemr.vandanapublications.com/index.php/ijemr/article/view/1565/1436)]                                                |           |     ✔️     |



## Tackling Information Embedding
1. Mercan Topkara, Cuneyt M Taskiran, and Edward J Delp III. Natural language watermarking. In Security, Steganography, and Watermarking of Multimedia Contents VII, volume 5681, pages 441–452. SPIE, 2005. [[paper](https://www.cerias.purdue.edu/tools_and_resources/bibtex_archive/archive/PSI000441.pdf)]

2. Umut Topkara, Mercan Topkara, and Mikhail J Atallah. The hiding virtues of ambiguity: quantifiably resilient watermarking of natural language text through synonym substitutions. In Proceedings of the 8th workshop on Multimedia and security, pages 164–174, 2006. [[paper](http://umut.topkara.org/papers/ToToAt_MMSEC06.pdf)]

3. Xi Yang, Jie Zhang, Kejiang Chen, Weiming Zhang, Zehua Ma, Feng Wang, and Nenghai Yu. Tracing text provenance via context-aware lexical substitution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 11613–11621, 2022. [[paper](https://cdn.aaai.org/ojs/21415/21415-13-25428-1-2-20220628.pdf)]

4. Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu. Watermarking text generated by black-box language models. arXiv preprint arXiv:2305.08883, 2023. [[paper](https://arxiv.org/pdf/2305.08883)]

5. Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, and Jiaheng Zhang. Provably robust multi-bit watermarking for ai-generated text via error correction code. arXiv preprint arXiv:2401.16820, 2024. [[paper](https://arxiv.org/pdf/2401.16820)]

### Training-free

#### Logits Deviation
1. John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pages 17061–17084. PMLR, 2023. [[paper](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)]

2. Xuandong Zhao, Prabhanjan Vijendra Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. In International Conference on Learning Representations, pages 1–35, 2024. [[paper](https://openreview.net/pdf?id=SsmT8aO45L)]

#### Hash-based
1. Abe Bohan Hou, Jingyu Zhang, Tianxing He, Yichen Wang, YungSung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. arXiv preprint arXiv:2310.03991, 2023. [[paper](https://arxiv.org/pdf/2310.03991)]

2. Yihan Wu, Zhengmian Hu, Hongyang Zhang, and Heng Huang. Dipmark: A stealthy, efficient and resilient watermark for large language models. In International Conference on Learning Representations, pages 1–27, 2024. [[paper](https://openreview.net/pdf?id=FhZi7r4nzA)]

3. Abe Bohan Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, and Tianxing He. k-semstamp: A clustering-based semantic watermark for detection of machine-generated text. arXiv preprint arXiv:2402.11399, 2024. [[paper](https://arxiv.org/pdf/2402.11399)]

#### Message Decoding
1. Xuandong Zhao, Lei Li, and Yu-Xiang Wang. Permute-and-flip: An optimally robust and watermarkable decoder for llms. arXiv preprint arXiv:2402.05864, 2024. [[paper](https://arxiv.org/pdf/2402.05864)]

2. Scott Aaronson, Jiahui Liu, Qipeng Liu, Mark Zhandry, and Ruizhe Zhang. New approaches for quantum copy-protection. In Advances in Cryptology–CRYPTO 2021: 41st Annual International Cryptology Conference, CRYPTO 2021, Virtual Event, August 16–20, 2021, Proceedings, Part I 41, pages 526–555. Springer, 2021. [[paper](https://arxiv.org/pdf/2004.09674)]

###  Training-based
#### Message Encoding
1. Han Fang, Zhaoyang Jia, Hang Zhou, Zehua Ma, and Weiming Zhang. Encoded feature enhancement in watermarking network for distortion in real scenes. IEEE Transactions on Multimedia, 2022. [[paper](http://staff.ustc.edu.cn/~zhangwm/Paper/2022_19.pdf)]

2. Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, and Farinaz Koushanfar. Remark-llm: A robust and efficient watermarking framework for generative large language models. In USENIX Security Symposium, 2024. [[paper](https://arxiv.org/pdf/2310.12362)]

### Information Capacity

#### Multi-bit
1. KiYoon Yoo, Wonhyuk Ahn, Jiho Jang, and Nojun Kwak. Robust multi-bit natural language watermarking through invariant features. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2092–2115, 2023. [[paper](https://arxiv.org/pdf/2305.01904)]

2. Pierre Fernandez, Antoine Chaffin, Karim Tit, Vivien Chappelier, and Teddy Furon. Three bricks to consolidate watermarks for large language models. In 2023 IEEE International Workshop on Information Forensics and Security (WIFS), pages 1–6. IEEE, 2023. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10374576)]

3. Massieh Kordi Boroujeny, Ya Jiang, Kai Zeng, and Brian Mark. Multi-bit distortion-free watermarking for large language models. arXiv preprint arXiv:2402.16578, 2024. [[paper](https://arxiv.org/pdf/2402.16578)]

## Dataset collation

|   Datasets   |  Size   |               Data Description               |
| :----------: | :-----: | :------------------------------------------: |
| TuringBench [[paper](https://arxiv.org/pdf/2109.13296)] |   200,000   |                News articles                 |
|     HC3 [[paper]](https://arxiv.org/pdf/2301.07597)     | 37,175  |   Reddit, Wikipedia, medicine and finance    |
|    CHEAT  [[paper](https://arxiv.org/pdf/2304.12008)] | 35,304  |              Academic abstracts              |
| GPT-Sentinel [[paper](https://arxiv.org/pdf/2310.08903)] | 29,395  |                 OpenWebText                  |
|   MGTBench [[paper]](https://arxiv.org/pdf/2303.14822)  |  2,817  |         Containing three enhanced datasets, i.e., Essay, WP, and Reuters          |
|   HC3 Plus [[paper](https://arxiv.org/pdf/2309.02731)] | 214,498 | Summarization, translation, and paraphrasing |
|  MULTITuDE [[paper](https://arxiv.org/pdf/2310.13606)]  | 74,081  |                 MassiveSumm                  |
|      M4 [[paper](https://openreview.net/pdf?id=ZTF2mX-Mpwh)]     | 122,481 |                Multi-generator, Multi-domain, and Multi-lingual                 |
|      M4GT-Bench [[paper](https://arxiv.org/pdf/2402.11175)]     | 138,465 |                Multi-lingual, Multi-domain, and Multi-generator                 |
|      GTD*   [[paper](https://arxiv.org/pdf/2405.06652)] |  1,378  |        Open-source dataset, Unknown source        |
|      Reviews24*   [[paper]](https://journals.sagepub.com/doi/10.1177/0261927X231200201)   |   1200    |       Self-built hotel review dataset        |
|   OpenGen  [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/575c450013d0e99e4b0ecf82bd1afaa4-Paper-Conference.pdf)]  |  3,000  | 3,000 randomly selected two-sentence blocks  |
|    Alpaca  [[paper](https://github.com/tatsu-lab/stanford_alpaca)]  |   52,000    |    Used for question and answer task test    |
|      C4    [[paper](https://www.jmlr.org/papers/v21/20-074.html)] |   \-    |  English corpus, used for text generation tasks      |
|   C4 News   [[paper](https://arxiv.org/pdf/2401.16820)] |   \-    |                15GB news data                |
|    Grover  [[paper]](https://proceedings.neurips.cc/paper_files/paper/2019/file/3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf)  |   \-    |     Generated by the news generator Grover-Mega (1.5B)      |
|     XSum   [[paper]](https://arxiv.org/pdf/1808.08745)  |   226,711    |                BBC articles and accompanying single sentence summaries                 |
|     WikiText-2    [[paper]](https://arxiv.org/pdf/1609.07843) |   720    |  Wiki text for language model training and evaluation |
|     WikiText-103    [[paper]](https://arxiv.org/pdf/1609.07843) |   28,595    |  Wiki text for language model training and evaluation |
|     LWD  [[paper]](https://arxiv.org/pdf/2401.06712)  |   234,593    |  Text generated using Llama-2, GPT-4, and ChatGPT    |
|     AAC  [[paper]](https://arxiv.org/pdf/2401.06712)  |   1,259,286    |  ext generated by GPT-2 and OPT models    |
|   HEIs*   [[paper]](https://link.springer.com/article/10.1007/s10805-023-09492-6) |   \-    |  963 student submissions of essays, reports, and case studies   |
|   DAIGT [[paper]](https://arxiv.org/pdf/2403.13335) |   44,206   |   The ratio of human-written text to LLM-generated text is 2:1    |
|   Deepfake  [[paper]](https://arxiv.org/pdf/2403.13335)  |   1562    |  LLM-generated text encompassing broader domains  |
|   BookSum  [[paper]](https://arxiv.org/pdf/2105.08209)  |   \-    |  a collection of datasets for long-form narrative summarization  |
|   RealNews  [[paper]](https://arxiv.org/pdf/1910.10683)  |   \-    |  Filtering C4 to only include news content  |
|   Creative Writing [[paper]](https://openreview.net/pdf?id=XnyZfCerSX)|   7000    |  Creative writing based on community tips  |
|   Student Essay [[paper]](https://openreview.net/pdf?id=XnyZfCerSX)|   7000    |  Essays based on the British Academic Written English corpus  |
|   News [[paper]](https://openreview.net/pdf?id=XnyZfCerSX)|   7000    |  Based on the Reuters 50-50 authorship identification datase  |
|   Code  [[paper]](https://arxiv.org/pdf/2401.12970)|   328    |  GPT-written Python code detection with HumanEval dataset |
|   Yelp Review  [[paper]](https://arxiv.org/pdf/2401.12970)  |   4000    |  Raw Yelp reviews are used and a clean AI-generated review is generated via GPT-3.5-turbo |
|   ArXiv Paper [[paper]](https://arxiv.org/pdf/2401.12970)|   700    |  Contains 350 abstracts of ICLR papers from 2015 to 2021  |




## Citation
If you find this project useful in your research or work, please consider citing it:

```
@article{yang2024survey,
  title={The Imitation Game Revisited: A Comprehensive Survey on Recent Advances in AI-Generated Text Detection},
  author={Zhiwei Yang, Zhengjie Feng, Ruichi Nie, Hongrui Chen, Hanghan Zheng, and Huiru Lin},
  year={2024}
}
```


## Acknowledgements
Your contributions will be acknowledged. 

[Github Flavored Markdown](https://github.com/guodongxiaren/README)
