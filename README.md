# SuperAnimal pretrained pose estimation models for behavioral analysis

![dlczoo](https://user-images.githubusercontent.com/28102185/209353843-cabc66e4-ab19-49df-8d46-5f1ddc9b5abe.png)

Part of the [DeepLabCut Model Zoo Project](modelzoo.deeplabcut.org)


## Figures and Data

Figures and data supporting Ye et al. 2023 (under revision).

- iRodent: a new dataset of rodents in the wild. Used here for out-of-domain testing. See https://zenodo.org/record/8250392 for data and more information.
- Models:
  - TopViewMouse, weights and model card are banked on HuggingFace: [https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)
  - Quadruped, weights and model card are banked on HuggingFace: [https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)


## Dependencies

```bash
pip install -r requirements.txt
```

## Repo organization

- ``data``: Folder to data files
- ``figures``: Rendered paper figures in `ipynb` format. 


## Citation

```
@article{ye2023superanimal,
      title={SuperAnimal pretrained pose estimation models for behavioral analysis}, 
      author={Shaokai Ye and Anastasiia Filippova and Jessy Lauer and Steffen Schneider and Maxime Vidal and Tian Qiu and Alexander Mathis and Mackenzie Weygandt Mathis},
      year={2023},
      eprint={2203.07436},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Keypoint model diagrams

As part of SuperAnimal, we released two major classes of pose models, TopViewMouse and Quadruped. Here are their keypoint guides:

<p align="center">
<img src="https://github.com/AdaptiveMotorControlLab/modelzoo-figures/blob/main/data/pose_skeleton_key_topviewmouse.png" width="33%">
</p>

<p align="center">
<img src="https://github.com/AdaptiveMotorControlLab/modelzoo-figures/blob/main/data/pose_skeleton_key_quadruped.png" width="43%">
</p>


