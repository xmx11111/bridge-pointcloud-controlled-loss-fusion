# Data Placement

Raw data are not included in this repository.

Suggested layout:

```text
data/
  BrPCD/
    official/
      prepared_20k_v2/
        train/
        val/
        test/
      projection_geom_512_v6_radius0/
  SemanticBridge/
    prepared_v1/
    projection_geom_512_radius0/
```

Download BrPCD from the official project:

`https://github.com/wxiong12/SEU-Digital-Bridge`

Download or prepare SemanticBridge from:

`https://github.com/mvg-inatech/3d_bridge_segmentation`

The exact author-side paths are visible in script defaults and command-line
arguments. Do not commit the downloaded datasets to this repository.
