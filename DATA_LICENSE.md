# Data attribution and modifications

The bundled hourly data and upload sample derive from:

Hebrail, G. and Berard, A. (2006). **Individual Household Electric Power Consumption** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

Source page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

UCI lists the dataset under **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/ . The source authors do not endorse this project.

Changes made: select global active power, parse local timestamps, aggregate minute readings to hourly means, require at least 45 valid readings per hour, mark low-coverage hours, and forward-fill at most three hours. The upload sample is a subset of this derivative. Evaluation challenge files additionally contain explicitly labelled artificial load changes. Original source values remain in `data/household_hourly.csv` after the stated preparation.

The DOI, source URL, original-archive hash, processed-data hash, counts and transformation notes are preserved in `data/provenance.json`. Keep this attribution and the modification notice when redistributing data derivatives.
