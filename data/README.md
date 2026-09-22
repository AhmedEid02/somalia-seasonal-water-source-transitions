# Data access

This repository does not include raw SIHBS 2022 microdata because the files are third-party data produced by the Somalia National Bureau of Statistics (SNBS).

## Official source

Somalia National Data Archive — Somalia Integrated Household Budget Survey (SIHBS) 2022:
https://microdata.nbs.gov.so/index.php/catalog/59

Reference ID: `DDI-SOM-SNBS-SIHBS-2022-v02`

## Files required by the reproduction script

Place these files in one local directory:

- `hh(2).dta`
- `hhm(2).dta`
- `agland(2).dta`
- `crops(2).dta`
- `livestock_own(1).dta`

Then provide that directory to `--data-dir` when running `analysis/reproduce_analysis.py`.

## Data availability statement for the manuscript

The data analyzed in this study are third-party data from the 2022 Somalia Integrated Household Budget Survey, produced by the Somalia National Bureau of Statistics. The author does not own the microdata and therefore does not redistribute them. Researchers can obtain the same data from the Somalia National Data Archive under the applicable SNBS access conditions. The analysis code and non-identifiable derived outputs are provided with the submission reproducibility materials.
