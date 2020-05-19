# SANS2020HackFest
SANS2020 HackFest Presentation

Software bug  recently submitted: oss-sec: CVE-2020-9391: Ignoring the top byte of addresses in brk causes heap corruption (AArch64)


Title: Automated detection of software vulnerabilities using Deep-learning

Abstract - The automatic detection of software vulnerabilities is an important security research problem. However, existing solutions are subjective to the expertise of humans who manually define features and often miss many vulnerabilities (i.e., incurring high false-negative rate). This presentation showcases the design and implementation of deep learning-based vulnerability detection systems to relieve human experts from the tedious and subjective task of manually defining features as well as to produce more effective vulnerability detection systems. The vulnerabilities that are detected are buffer errors and resource management errors in software.  An approach called code gadgets [1] is used, which represents software programs and then transforms them into vectors. A code gadget is the number of lines of code that are semantically related to each other. The approach then demonstrates the identification of vulnerabilities in different software products. The attendees will learn how deep-learning methods are more than just an improvement over the traditional vulnerability detection systems. They will understand the end-to-end implementation and be able to replicate it at their workplace.
 
Outline 
1.     Preparing the environment by providing instructions (in Linux, windows, and mac) for installing software and dependencies: pandas, gensim, Keras, TensorFlow, and sklearn packages.
a.     This should be announced to the participants before they come for the training session.
2.     Git repo installation instructions of the dataset (list of various instructions) that captures the vulnerable software (shared apriori) – buffer error and resource management errors. Example:
a.     1 CVE-2010-1444/vlc_media_player_1.1.0_CVE-2010-1444_zipstream.c cfunc 449
b.     ZIP_FILENAME_LEN, NULL, 0, NULL, 0 )
c.      char *psz_fileName = calloc( ZIP_FILENAME_LEN, 1 );
d.     if( unzGetCurrentFileInfo( file, p_fileInfo, psz_fileName,
e.     vlc_array_append( p_filenames, strdup( psz_fileName ) );
f.      free( psz_fileName );
g.     0
3.     Deep Learning algorithms that will be used – review w/ examples (10 minutes)
a.     Algorithms -   BLSTM, RNN, Encoding, Vector Representation
b.     Background links for a quick study of these algorithms will be provided a-priori as it is too much to cover when we’re focusing on the application
4.     Train and Test Deep learning model on the dataset provided
a.     Git downloadable python code for this
b.     Expected output – screenshots, live demo
5.     Prediction based on output: Vulnerable/ Clean


