# GAN TTS
we propose a framework for Test-to-speech (TTS) using the Generative adversarial networks (GANs) for the Low Resource Indian Languages. Deep neural networks are emerging as powerful techniques for the synthetic production of speech, however, the quality is less natural as compared to an original speech waveform. Over- smoothing is one of the major problems which degrades the quality of speech synthesizers. We introduced the GANs, in this research paper for the low resource Indian languages. GANs comprises of two neural networks, a discriminator to distinguish natural and generated samples, and a generator to deceive the discriminator. This work is focused to design the discriminator which is trained to identify the difference between the original speech and synthesized speech parameters, and generator models the acoustic parameters from the given speech waveform. It results in reducing the distribution of the difference between the original speech and synthesized speech waveforms for the Text-to-Speech (TTS) system for the low resource Indian languages such as Hindi. The presented novel method efficiently builds the synthetic quality of speech for the low resource Indian language



## Requirements

- [PyTorch](http://pytorch.org/) >= v0.2.0
- [TensorFlow](https://www.tensorflow.org/) (just for `tf.contrib.training.HParams`)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii)
- https://github.com/taolei87/sru (if you want to try SRU-based models)
- Python

## Installation

Please install PyTorch, TensorFlow and SRU (if needed) first. Once you have those, then


should install all other dependencies.

## Repository structure

- **gantts/**: Network definitions, utilities for working on sequence-loss optimization.
- **prepare_features_vc.py**: Acoustic feature extraction script for voice conversion.
- **prepare_features_tts.py**: Linguistic/duration/acoustic feature extraction script for TTS.
- **train.py**: GAN-based training script. This is written to be generic so that can be used for training voice conversion models as well as text-to-speech models (duration/acoustic).
- **train_gan.sh**: Adversarial training wrapper script for `train.py`.
- **hparams.py**: Hyper parameters for VC and TTS experiments.
- **evaluation_vc.py**: Evaluation script for VC.
- **evaluation_tts.py**: Evaluation script for TTS.

Feature extraction scripts are written for CMU ARCTIC dataset, but can be easily adapted for other datasets.


### Voice conversion (en)

`vc_demo.sh` is a `clb` to `clt` voice conversion demo script. Before running the script, please download wav files for `clb` and `slt` from [CMU ARCTIC](http://festvox.org/cmu_arctic/) and check that you have all data in a directory as follows:

```
> tree ~/data/cmu_arctic/ -d -L 1
/home/ryuichi/data/cmu_arctic/
├── cmu_us_awb_arctic
├── cmu_us_bdl_arctic
├── cmu_us_clb_arctic
├── cmu_us_jmk_arctic
├── cmu_us_ksp_arctic
├── cmu_us_rms_arctic
└── cmu_us_slt_arctic
```

Once you have downloaded datasets, then:

```
./vc_demo.sh ${experimental_id} ${your_cmu_arctic_data_root}
```

e.g.,

```
 ./vc_demo.sh vc_gan_test ~/data/cmu_arctic/
```

Model checkpoints will be saved at `./checkpoints/${experimental_id}` and audio samples
are saved at `./generated/${experimental_id}`.

### Text-to-speech synthesis (en)

`tts_demo.sh` is a self-contained TTS demo script. The usage is:

```
./tts_demo.sh ${experimental_id}
```

This will download `slt_arctic_full_data` used in Merlin's demo, perform feature extraction, train models and synthesize audio samples for eval/test set. `${experimenta_id}` can be arbitrary string, for example,

```
./tts_demo.sh tts_gan_test ~/data/cmu_arctic/
```


Model checkpoints will be saved at `./checkpoints/${experimental_id}` and audio samples
are saved at `./generated/${experimental_id}`.

## Hyper paramters

See ``hparams.py``.

## Monitoring training progress

```
tensorboard --logdir=log
```

## References

- [Yuki Saito, Shinnosuke Takamichi, Hiroshi Saruwatari, "Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks", arXiv:1709.08041 [cs.SD], Sep. 2017](https://arxiv.org/abs/1709.08041)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Training algorithm to deceive anti-spoofing verification for DNN-based text-to-speech synthesis," IPSJ SIG Technical Report, 2017-SLP-115, no. 1, pp. 1-6, Feb., 2017. (in Japanese)](http://sython.org/papers/SIG-SLP/saito201702slp.pdf)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Voice conversion using input-to-output highway networks," IEICE Transactions on Information and Systems, Vol.E100-D, No.8, pp.1925--1928, Aug. 2017](https://www.jstage.jst.go.jp/article/transinf/E100.D/8/E100.D_2017EDL8034/_article)
- https://www.slideshare.net/ShinnosukeTakamichi/dnnantispoofing
- https://www.slideshare.net/YukiSaito8/Saito2017icassp


