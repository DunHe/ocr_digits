# ocr_digits

### 环境搭建

pip install -r requirements.txt

添加下面代码到~/.bashrc中:
export PYTHONPATH=/path/to/ocr_digits/python:$PYTHONPATH
export PATH=/path/to/ocr_digits/bin:$PATH

### 运行

bin/train_digits
bin/reco_digits

```bash
./train_digits --dataset ../../ocr_datasets/mnist.pkl.gz
./reco_digits --image ../example/data/reco.png
```
