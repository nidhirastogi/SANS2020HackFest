# SANS 2020 Summit talk - Deep Learning

Title: Automated detection of software vulnerabilities using Deep-learning

### Abstract
The automatic detection of software vulnerabilities is an important security research problem. However, existing solutions are subjective to the expertise of humans who manually define features and often miss many vulnerabilities (i.e., incurring high false-negative rate). This presentation showcases the design and implementation of deep learning-based vulnerability detection systems to relieve human experts from the tedious and subjective task of manually defining features as well as to produce more effective vulnerability detection systems. The vulnerabilities that are detected are buffer errors and resource management errors in software.  An approach called code gadgets [1] is used, which represents software programs and then transforms them into vectors. A code gadget is the number of lines of code that are semantically related to each other. The approach then demonstrates the identification of vulnerabilities in different software products. The attendees will learn how deep-learning methods are more than just an improvement over the traditional vulnerability detection systems. They will understand the end-to-end implementation and be able to replicate it at their workplace.
 
### Outline
 1. Preparing the environment by providing instructions (in Linux, windows, and mac) for installing software and dependencies: pandas, gensim, Keras, TensorFlow, and sklearn packages.
 2.  Git repo installation instructions of the dataset (list of various instructions) that captures the vulnerable software (shared apriori) – buffer error and resource management errors. 
 3. C++ code containing vulnerabilities
 4. Deep Learning algorithm that will be used BLSTM
 5. Train and Test Deep learning model on the dataset provided
 6. Git downloadable python code for this 
 7. Expected output –screenshots, recorded demo
 8. Prediction based on output: Vulnerable/ Clean

### References
 1. [VulDeePecker: A Deep Learning-Based System for Vulnerability Detection](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-2_Li_paper.pdf)
2. [Automated Vulnerability Detection in Source Code Using Deep Learning](https://arxiv.org/pdf/1807.04320.pdf)
3. [Github - VulDeePecker](https://github.com/CGCL-codes/VulDeePecker)
4. [Github - VDPython](https://github.com/johnb110/VDPython)
