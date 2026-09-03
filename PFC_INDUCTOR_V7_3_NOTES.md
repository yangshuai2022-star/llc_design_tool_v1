# V7.3 PFC Inductor Design

## Default core

Magnetics High Flux toroid, Core Data 254. Built-in default material grade: 60 µ.

Core table source:
https://www.mag-inc.com/zh-cn/products/powder-cores/high-flux-cores

Material curve source:
https://www.mag-inc.com/zh-cn/products/powder-cores/high-flux-cores/high-flux-material-curves

Built-in Core Data 254 geometry:
- Le = 98.4 mm
- Ae = 110.6 mm²
- Ve = 10880 mm³
- OD = 40.77 mm
- ID = 23.32 mm
- HT = 15.37 mm

The 60 µ grade uses AL = 81 nH/T² on Core Data 254.

## Manufacturer fits

DC-bias permeability fit: `%ui = 1 / (a + b*H^c)`, H in Oe.
Core-loss density fit: `Pv = a*B^b*f^c`, B in T, f in kHz, Pv in mW/cm³.

Only permeability grades with both a Core Data 254 AL value and complete published DC-bias/core-loss coefficients are selectable in the automatic model.

## Copper model

Default winding is enamelled round copper. Copper loss is hot DC I²R only. Skin-effect and proximity-effect loss are intentionally excluded in V7.3 per the design requirement.
