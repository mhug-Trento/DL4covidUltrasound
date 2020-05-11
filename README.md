# DL4covidUltrasound

<p align="center">
    <img src="./imgs/teaser_rebuttal.png"/> <br />
    <em> 
    Figure 1. Overview of the different tasks considered in this work. Given a LUS image sequence, we propose approaches for: (orange) predictionof the disease severity score for each input frame and weakly supervised localization of pathological patterns; (pink) aggregation of frame-levelscores for producing predictions on videos; (green) estimation of segmentation masks indicating pathological artifacts.
    </em>
</p>

## 2. Proposed Methods

Our method foresees two main components:

- A frame-based predictor exploiting STN [1] to recognize the promising region crop in the input data.
- The aggregation of the frame-based predictions.

### 2.1 Frame-based Score Prediction

<p align="center">
    <img src="./imgs/stn_predictor.png" width="500"> <br />
    <em> 
    Figure 2. Illustration of the architecture for frame-based score predic- tion. An STN modeled by Φstn predicts two transformations θ1 and θ2 which are applied to the input image producing two transformed versions x1 and x2 that localize pathological artifacts. The feature extractor Φcnn is applied to x1 to generate the final prediction.
    </em>
</p>



### 2.2 Video-based Score Prediction

Here is an example of video-based predictions exploiting our aggregation strategy of frame-based predictions.

<p align="center">
    <img width="99%" src="imgs/video_prediction.gif" />
    <em> 
    Video 1. Visualization of the results of our video-based score aggregation method.
    </em>

</p>


## 3. Results



## 4. Installation



### Requirements

### Evaluation

### Train your own model

## 5. Manuscript

The pre-print copy of our manuscript can be found at [ppp]()

## 6. Citation

Please cite our paper if you find the work useful:

    @article{DL4covidUltrasound,
    title={TTT},
    author={aaa},
    Journal = {arXiv},
    year={2020}
    }

 
## 7. Acknowledgements

We thanks [yyy]() for their contributions.