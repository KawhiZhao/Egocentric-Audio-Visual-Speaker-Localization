# Egocentric Audio Visual Speaker Localization

### The simulated dataset Ego-AVST is available on request

## Experiments on Ego-AVST

1. Split the dataset into chunks so that every chunk in a subfolder containing image, audio, gt, bbox, depth image

    `cd data` \
    Using the function `split` in `split_avst.py`

2. Calculate GCCPHAT for each audio clip in each chunk

    Using the function `test` in `gccphat_avst.py`

3. Train

    `python main_avst.py`

4. Evaluation

    Put the path of pretrained model in `args.load_path` in `main_avst.py`

    Comment the training code \
    `trainer.fit(model, data_module)`

    Uncomment the code block\
    `checkpoint = torch.load(args.load_path)`\
    `model.load_state_dict(checkpoint['state_dict'])`\
    `model.eval()`
    

## Experiments on EasyCom

1. Split the data into training set and test set. Session 1, 2, 3 will be in the test set and the remaining sessions will be in the training set. The whole video list can be found in `video.txt` under the `EasyCom` folder.

2. Split the dataset into chunks so that every chunk in a subfolder containing image, audio, gt

    `cd data` \
    Using the function `split` in `split.py`

3. Calculate GCCPHAT for each audio clip in each chunk

    Using the function `test` in `gccphat.py`

4. Train

    `python main.py`

5. Evaluation

    Put the path of pretrained model in `args.load_path` in `main.py`

    Comment the training code \
    `trainer.fit(model, data_module)`

    Uncomment the code block\
    `checkpoint = torch.load(args.load_path)`\
    `model.load_state_dict(checkpoint['state_dict'])`\
    `model.eval()`

## Reference

    EasyCom Dataset https://github.com/facebookresearch/EasyComDataset
    Transformer Block: https://github.com/jadore801120/attention-is-all-you-need-pytorch
    GCCPHAT Calculation: https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
    Lightning: https://github.com/miracleyoo/pytorch-lightning-template
