# Flat-SVM
A flat multi-label SVM text classifier implemented with python and Scikit-lear library. This code is part of <a href="https://atrium.lib.uoguelph.ca/xmlui/handle/10214/14251">this</a> work. 

# Dataset
The dataset is a CSV file that has occupation titles and their <a href="https://www.bls.gov/soc/
">Standard Occupational Classification (SOC)</a>. I have collected the data from ONET website, US Census data, and SOC website. The occupation titles can have multiple SOC codes. The reasons for multiple codes are explained in the <a href="https://atrium.lib.uoguelph.ca/xmlui/handle/10214/14251"> this work</a>. Since SOC has four levels of hierarchy, we implement a flat multi-label classifier for each level.

