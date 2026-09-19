# Data Sources and Preparation

## BrPCD

Source repository:

`https://github.com/wxiong12/SEU-Digital-Bridge`

The paper uses the official `prepared_20k_v2` split:

- 53 training scenes
- 2 validation scenes
- 11 test scenes
- primary evaluation family: `c-bridge4`

Prepared scenes contain `xyz`, `rgb`, `labels`, `center`, `extent`, and
`scale`. Class IDs are girder, pier, and deck in that order.

Preparation entry point:

```bash
python scripts/brpcd/prepare_brpcd.py --help
```

Audit entry point:

```bash
python scripts/brpcd/audit_brpcd.py --help
```

## SemanticBridge

Project code and data access:

`https://github.com/mvg-inatech/3d_bridge_segmentation`

The study uses real TLS instances. The original class IDs are mapped as
follows:

| Original class | ID | Study class |
|---|---:|---|
| superstructure | 4 | girder |
| abutment | 3 | pier |
| pillar | 8 | pier |
| top surface | 5 | deck |

Ground, high vegetation, railing, traffic sign, and unlabelled points are
outside the three-class evaluation.

SemanticBridge is used only for external trend analysis. Absolute accuracies
are not comparable to BrPCD because acquisition modes, scene scale, and class
definitions differ.
