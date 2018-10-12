# Flat Multi-label Text Classification
A flat multi-label text classifier implemented with SVM, Naive Bayes, Maximum Entropy. This code is part of <a href="https://atrium.lib.uoguelph.ca/xmlui/handle/10214/14251">this</a> work. 

# Requierments:
Python 2.7

Scikit-Learn

# Dataset
The dataset is a CSV file that has occupation titles and their <a href="https://www.bls.gov/soc/
">Standard Occupational Classification (SOC)</a>. I have collected the data from ONET website, US Census data, and SOC website. The occupation titles can have multiple SOC codes. The reasons for multiple codes are explained in the <a href="https://atrium.lib.uoguelph.ca/xmlui/handle/10214/14251"> this work</a>. Since SOC has four levels of hierarchy, we implement a flat multi-label classifier for each level.

