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



