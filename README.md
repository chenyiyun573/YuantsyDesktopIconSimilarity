
Install dependency packages with the following command:

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

20240624 0019
The deep learning one always fails due to the downloading of the model paras:
```
(.venv) /Users/chenyiyun/Desktop/YuantsyDesktopIconSimilarity/.venv/bin/python /Users/chenyiyun/Desktop/YuantsyDesktop
IconSimilarity/main.py
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
 72417280/553467096 ━━━━━━━━━━━━━━━━━━━━ 1:45 0us/step
```
So, I commented it out. 

This repo is not completed. Next todo, compare and explain these similarities in the README.md.