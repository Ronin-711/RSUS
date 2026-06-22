# RSUS

This repository is for the official code release of **RSUS**, a dedicated upsampling method for remote sensing image semantic segmentation.

RSUS supports arbitrary-scale upsampling and can help different segmentation models improve their segmentation performance.

The paper has been submitted to **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**.

Update 2026/6/22
We've published the source code for RSUS, which can be found in RSUS_code.py
Using it is very simple, as shown in the following example

    up = RSUS(
       in_ch_high=64,
        out_ch=64,
        scale=4
    ).to(device)
    up.eval()

