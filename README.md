# RSUS

This repository is for the official code release of **RSUS**, a dedicated upsampling method for remote sensing image semantic segmentation.

RSUS supports arbitrary-scale upsampling and can help different segmentation models improve their segmentation performance.
![RSUS Framework](figures/RSUS.jpg)

The paper has been submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**.
Update 2026/7/13
Our article has been accepted by IEEE TGRS！and DOI:10.1109/TGRS.2026.3714102

Update 2026/6/22
We've published the source code for RSUS, which can be found in RSUS_code.py
Using it is very simple, as shown in the following example

    up = RSUS(
       in_ch_high=64,
        out_ch=64,
        scale=4
    ).to(device)
    up.eval()

In addition, we also provide the code of CARAFE++ for reference：Digital Object Identifier no. 10.1109/TPAMI.2021.3074370
The persistent homology method used in this paper is also disclosed here, which can be used as a reference for topology analysis applications. The detailed code can be found in PH.py


The XiNing dataset section of the paper is available at:
 https://pan.baidu.com/s/1kaBKFEKsmZWZPcCeIiVwkQ?pwd=vfdc 提取码: vfdc
