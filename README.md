# Deep learning for classification and localization of COVID-19 markers in point-of-care lungultrasound

<p align="center">
    <img src="./imgs/teaser_rebuttal.png"/> <br />
    <em> 
    Figure 1. Overview of the different tasks considered in this work.
    </em>
</p>

> **Deep learning for classification and localization of COVID-19 markers in point-of-care lungultrasound**<br>
> [Subhankar Roy](https://github.com/roysubhankar), [Willi Menapace](https://github.com/willi-menapace), Sebastiaan Oei, Ben Luijten, [Enrico Fini](https://github.com/DonkeyShot21), [Cristiano Saltori](https://github.com/saltoricristiano), Iris Huijben, Nishith Chennakeshava, Federico Mento, Alessandro Sentelli, Emanuele Peschiera, Riccardo Trevisan, Giovanni Maschietto, Elena Torri, Riccardo Inchingolo, Andrea Smargiassi, Gino Soldati, [Paolo Rota](https://github.com/paolorota), [Andrea Passerini](https://github.com/andreapasserini), Ruud J.G. van Sloun, [Elisa Ricci](http://elisaricci.eu/), Libertario Demi<br>
> In IEEE Transactions on Medical Imaging.<br>

> Paper: arxiv-link<br>

> **Abstract:** *Deep learning (DL) has proved successful inmedical  imaging  and,  in  the  wake  of  the  recent  COVID-19 pandemic, some works have started to investigate DL-based  solutions  for  the  assisted  diagnosis  of  lung  diseases. While existing works focus on CT scans, this paper studies  the  application  of  DL  techniques  for  the  analysisof  lung  ultrasonography  (LUS)  images.  Specifically,  we present  a  novel  fully-annotated  dataset  of  LUS  images collected from several Italian hospitals, with labels indicating the degree of disease severity at a frame-level, video-level,  and  pixel-level  (segmentation  masks).  Leveraging these data, we introduce several deep models that address relevant  tasks  for  the  automatic  analysis  of  LUS  images. In  particular,  we  present  a  novel  deep  network,  derived from Spatial Transformer Networks, which simultaneously predicts  the  disease  severity  score  associated  to  a  input frame  and  provides  localization  of  pathological  artefacts in a weakly-supervised way.  Furthermore, we introduce a new method based on uninorms for effective frame score aggregation  at  a  video-level.  Finally,  we  benchmark  state of the art deep models for estimating pixel-level segmentations of COVID-19 imaging biomarkers. Experiments on the proposed  dataset  demonstrate  satisfactory  results  on  all the considered tasks, paving the way to future research on DL for the assisted diagnosis of COVID-19 from LUS data.*

## 2. Proposed Methods

Our method foresees two main components:

- A frame-based predictor exploiting a novel deep architecture based on STN [1] that provides the disease severity score and a weakly-supervised localization of pathological artefacts.
- A video-based score predictor based on uninorms that performs aggregation of the frame-based scores.

### 2.1 Frame-based Score Prediction

<p align="center">
    <img src="./imgs/stn_predictor.png" width="600"> <br />
    <em> 
    Figure 2. Illustration of the proposed Reg-STN architecture for frame-based score prediction.
    </em>
</p>

## 3. Results

### 3.1 Frame-based Score Prediction

<p align="center">
    <img src="./imgs/frame_results.png" width="800"> <br />
    <em> 
    Table 1. F1 scores (%) for the frame-based classification under different evaluation settings explained in the paper.
    Best and second best F1 scores (%) are in bold and underlines, respectively.
    </em>
</p>

<p align="center">
    <img src="./imgs/stn_loc.png" width="700"> <br />
    <em> 
    Figure 3. Examples of weakly-supervised localization results of pathological artefacts produced by the Reg-STN network.
    </em>
</p>

<p align="center">
    <img width="99%" src="imgs/example_video.gif" />
    <em> 
    Video 1. Examples of frame-based predictions produced by our proposed Reg-STN model.
    </em>

</p>

### 3.2 Video-based Score Prediction

<p align="center">
    <img src="./imgs/video_results.png" width="400"> <br />
    <em> 
    Table 2. Mean and standard deviation of weighted F1 score, precision and recall for the proposed video-based classification method and baselines.
    </em>
</p>

### 3.2 Frame-based Segmentation

Results coming soon

## 4. Installation

Coming soon ...

### Requirements

We try to minimize requirements. Assuming you are using `Python 3.7+`, you can install requirements:
```
python -m pip install -r requirements.txt
```

## 5. Dataset
Dataset information coming soon!

## 6. Usage

### Evaluation

Coming soon ...

### Training

Coming soon ...

#### Video-based Score Prediction

The video-based score predictor can be trained by running the following command inside the `video_score_predictor` directory

```
python aggregator.py --use_sord --setting=kfolds --lr=0.01 --tnorm=product --zero_score_gap=0.5 --loss=ce --epoch=30 --earlystop=last --init_neutral=0. --lr_gamma=1 --off_diagonal=min --testfile '' --expname=<experiment_name> 'data/frame_predictions.pkl' 'data/video_annotations.xlsx' 'data/video_annotations_to_video_names.xlsx' <output_path>
```

## 7. Citation

Please cite our paper if you find the work useful:

    @article{DL4covidUltrasound,
    title={TTT},
    author={aaa},
    Journal = {arXiv},
    year={2020}
    }

 
## 8. Acknowledgements

We thank the Caritro Deep Learning Lab of [ProM Facility](https://promfacility.eu/#/) who made available their GPUs for the current work. We also thank [Fondazione VRT](https://www.fondazionevrt.it/) for financial support [COVID-19 CALL 2020 Grant #1]

## 9. License

This work is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
