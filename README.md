# LET
Source code of AAAI2021 paper "[LET: Linguistic Knowledge Enhanced Graph Transformer for Chinese Short Text Matching](https://ojs.aaai.org/index.php/AAAI/article/view/17592)".

## Requirements
* `python`: 3.7.5
* `mxnet-cu100`: 1.5.1.post0
* `gluonnlp`: 0.8.0
* `jieba`: 0.39
* `thulac`: 0.2.1
* `pkuseg`: 0.0.22



## Training
Before training, please contact the author of [BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm) and [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) dataset to download them.
Then, you need to process the data to get the same format as the file [`data/LCQMC/train.json`](https://github.com/lbe0613/LET/blob/main/data/LCQMC/train.json).

```bash
$ python utils/preprocess.py -i data/LCQMC/train.txt -o data/LCQMC/train.json
```

Train the model:
```bash
$ python train_sememe.py -c config/train_sememe_LCQMC.conf
```

The models trained by us can be downloaded from [LET_BQ](https://pan.baidu.com/s/13FS0wg2vP8XGlVcCYl_iSg) (password: aif3) and [LET_LCQMC](https://pan.baidu.com/s/1jQEidBRYo519j2NGnLJlBQ) (password: udbv).

## Cite
If you find our code is useful, please cite:
```
@inproceedings{lyu2021let,
  title={LET: Linguistic Knowledge Enhanced Graph Transformer for Chinese Short Text Matching},
  author={Lyu, Boer and Chen, Lu and Zhu, Su and Yu, Kai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={15},
  pages={13498--13506},
  year={2021}
}
```



