# Benchmark Results

| Architecture      | Run               | Source      |   Input Size |   Params (M) | Pretrained   |   Best Acc (%) |   Best Epoch |   Final Val Acc (%) |   Final Val Loss |   Epochs |
|:------------------|:------------------|:------------|-------------:|-------------:|:-------------|---------------:|-------------:|--------------------:|-----------------:|---------:|
| tv_convnext       | tv_convnext       | TorchVision |          128 |      27.8300 | yes          |        99.0300 |          114 |             98.8700 |           0.1389 |      183 |
| tv_efficientnetv2 | tv_efficientnetv2 | TorchVision |          128 |      20.1900 | yes          |        98.6800 |          188 |             98.5300 |           0.2523 |      235 |
| timm_resnet       | timm_resnet       | TIMM        |          224 |      11.1800 | yes          |        97.3000 |          149 |             97.3000 |           0.2899 |      150 |
| own_wrn           | own_wrn           | Own         |           32 |       2.7500 | no           |        96.5900 |          294 |             96.5000 |           0.2857 |      300 |
| own_vgg           | own_vgg           | Own         |           32 |      19.1800 | no           |        96.3800 |          186 |             96.2600 |           0.2124 |      200 |
| own_deit          | own_deit          | Own         |           32 |       2.8600 | no           |        93.1500 |          300 |             93.1500 |           0.2533 |      300 |
| own_vit           | own_vit           | Own         |           32 |       2.6900 | no           |        89.5500 |          243 |             89.2600 |           0.4295 |      300 |
