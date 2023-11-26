# Deep learning-based electrocardiogram rhythm and beat features for heart abnormality classification

> Annisa Darmawahyuni, Siti Nurmaini, Muhammad Naufal Rachmatullah, Bambang Tutuko, Ade Iriani Sapitri, Firdaus Firdaus, 
Ahmad Fansyuri and Aldi Predyansyah

## Introduction

- Electrocardiogram (ECG) measures the electricity of the heart, by analyzing each electrical signals, it is possible to 
detect some abnormalities.
- While the acquisition of ECG recordings follows standardized procedures, diverse human interpretations, influenced by 
varying levels of experience, expertise, and nuances, have prompted the utilization of computer-generated interpretations, 
albeit limited by predetermined rules and feature recognition algorithm constraints (Siontis et al., 2021).
- Utilizing rhythm and beat features, ECG signal processing, particularly through the application of deep learning as an 
artificial intelligence approach, has become widely employed for analyzing heart abnormalities, providing effective 
diagnosis for conditions such as arrhythmia, heart failure, myocardial infarction, left ventricular hypertrophy, valvular 
heart disease, age, and sex based on ECG signals alone (Darmawahyuni et al., 2021; Makimoto et al., 2020; Attia et al., 
2019; Hannun et al., 2019; Kwon et al., 2020a; Kwon et al., 2020b; Yildirim, 2018; LeCun, Bengio & Hinton, 2015).
- In prior research, ECG signal classification focusing on heart rhythm incorporates various features, including the 
morphology of the ECG signal such as ST-elevation, depression, T-wave abnormalities, and pathological Q-waves (Ansari et 
al., 2017); additionally, the automatic detection of heart abnormalities has been pursued using ECG rhythm features like 
the R-R interval, S-T interval, P-R interval, and Q-T interval over the past decade (Gopika et al., 2020).
- For the classification of heartbeats, the ECG pattern may exhibit similarity among patients with different heartbeats 
and variation within the same patient at different times, rendering ECG-based heartbeat classification essentially a 
challenge of temporal pattern recognition and classification (Zubair, Kim & Yoon, 2016; Dong, Wang & Si, 2017).
- Beyond the inherent challenge of analyzing ECG signal patterns, characterized by small amplitudes and short durations 
measured in millivolts and milliseconds, and subject to substantial inter- and intra-observer variability (Lih et al., 
2020), the time-consuming nature of manually analyzing thousands of ECG signals increases the risk of misreading vital 
information. Addressing these challenges, automated diagnostic systems leveraging computerized recognition of heart 
abnormalities based on rhythm or beat offer a potential solution, with the prospect of becoming a standard procedure in 
clinicians' classification of ECG recordings.
- This study proposes a unified deep learning (DL) architecture for ECG pattern classification, incorporating both rhythm 
and heartbeat features in a single framework. The chosen DL model, a one-dimensional convolutional neural network (1D-CNN), 
has demonstrated promising results in previous works (Nurmaini et al., 2020; Tutuko et al., 2021), achieving high accuracy 
by integrating feature extraction, dimensionality reduction, and classification techniques. Notably, 1D-CNN has shown 
success in ECG classification for both rhythm and beat, with accuracies ranging from 93.53% to 97.4% and 92.7% to 96.4%, 
respectively, as reported in various studies (Acharya et al., 2017; Wang, 2020; Zubair, Kim & Yoon, 2016; Kiranyaz, Ince 
& Gabbouj, 2015). This underscores the effectiveness of 1D-CNN in pattern recognition for ECG analysis.
- This study makes the following novel contributions:
	- Proposes the generalization framework of deep learning for ECG signal classification with high accuracy in intra- 
	and inter-patients’ scenario;
	- Develops a single DL-architecture for classifying ECG signal pattern based on both of rhythm and beat feature with 
	simple segmentation;
	- Validates the proposed framework with five public ECG datasets that have different frequency sampling with massive 
	data; and
	- Experiment with 24-class abnormalities found in the ECG signal, consisting of nine-class of ECG pattern based on 
	rhythms feature and 15-class of ECG pattern based on beats feature.

## Material and Methods

### Data Preparation

- In this study, the PhysioNet public dataset (Goldberger et al., 2000), containing ECG data from both healthy volunteers 
and patients with various heart diseases, is utilized, with assurance that the information has been ethically disclosed 
by a third party unrelated to the current research.
- To perform ECG signal pattern recognition, two segmentation processes, rhythm and beat, are employed, with the 
experimental databases divided into two cases:
	- for **ECG rhythm classification**, datasets such as PTB Diagnostic ECG (PTB DB), BIDMC Congestive Heart Failure (CHF), 
	China Physiological Signal Challenge 2018, and MIT-BIH Normal Sinus Rhythm are utilized (Bousseljot, Kreiseler & Schnabel, 
	1995; Baim et al., 1986; Liu et al., 2018; Goldberger et al., 2000); and 
	- for **ECG beat classification**, the MIT-BIH Arrhythmia Database is used (Moody & Mark, 2001).

### Proposed Methodology for 1D-CNN

- In this study, a novel methodology is presented, employing a single deep learning (DL) architecture to classify 25 
classes of ECG signal patterns based on both rhythm and beat features, enabling the simultaneous processing of beats and 
rhythms within a unified DL framework. The proposed approach involves ECG signal denoising, beat and rhythm segmentation, 
and classification, allowing the detection of pattern abnormalities in both beats and rhythms using a single architecture. 
The methodology builds upon the generalized 1D-CNN architecture previously published in works by Nurmaini et al. (2020) 
and Tutuko et al. (2021).
- The methodology follows a standardized evaluation process with a clinical perspective, ensuring reproducibility and 
comparability of experiments through five main stages: 
	1. selection of ECG public databases; 
	2. pre-processing of ECG signals via discrete wavelet transforms to eliminate noise and artifacts; 
	3. segmentation of ECG signals into 2,700 and 252 nodes based on rhythm and beat, respectively; 
	4. utilization of 1D-CNN for feature extraction and classification, learning characteristics of rhythm and beat 
	episodes; and 
	5. evaluation of the proposed model on validation and testing data, assessing accuracy, sensitivity, specificity, 
	precision, and F1-score.

#### Database Selection

A total of 168,472 rhythm episodes and 110,082 beat episodes from single-lead ECG standard recordings, with varying 
signal lengths and frequency samplings (128, 250, 500, and 1000 Hz), were employed for training, validation, and testing 
using a 1D-CNN architecture to classify nine classes based on rhythm feature segmentation and 15 classes based on beat 
feature segmentation.

#### Pre-processing

The study addresses the issue of ECG signal corruption during acquisition by various artifacts and interference, such as 
muscle contraction, baseline drift, electrode contact noise, and power line interference (Sameni et al., 2007; Tracey & 
Miller, 2012; Wang et al., 2015). To enhance analysis and diagnosis accuracy, the study employs discrete wavelet transform 
(DWT) as a denoising technique, experimenting with different wavelet families, and concludes that the symlet wavelet 
yielded the best signal denoising results based on the highest signal-to-noise ratio (SNR).

#### ECG Signal Segmentation

The objective of ECG segmentation is to partition a signal into segments with similar statistical properties, such as 
amplitude, nodes, and frequency, as the presence, time, and length of each segment hold diagnostic and biophysical 
significance, and different sections of an ECG signal carry distinctive physiological meaning (Yadav & Ray, 2016), 
contributing to an accurate analysis of ECG signal segmentation. The process of ECG feature selection are described:

1. In the ECG rhythm segmentation process for the classification of nine classes of normal-abnormal ECG rhythms, features 
were generated for the entire ECG signal recordings at 2,700 nodes, with consideration for different frequency samplings 
(128, 250, 500, and 1,000 Hz) (Nurmaini et al., 2020). The 2,700-node segmentation ensured at least two R-R intervals 
between consecutive beats, and zero-padding was applied if the total nodes were less than 2,700.
2. ECG beat segmentation involves intercepting multiple nodes in a signal to identify both subsequent heartbeats and the 
waveforms within each beat, distinguishing between characteristics retrieved from a single beat, typically containing 
one R-peak, and features dependent on at least two beats, providing more information than a single R-peak (Qin et al., 
2017). The positions of the P-wave, QRS-complex, and T-wave in beat segmentation are intricately linked to the R-peak. 
According to Qin et al. (2017), Chang et al. (2012), and Nurmaini et al. (2019), the average ECG rhythm frequency falls 
between 60 and 80 beats per minute, with t1 duration of 0.25 s before the R-peak, t2 duration of 0.45s after the R-peak, 
resulting in a total length of 0.7s, equivalent to 252 nodes at a sampling frequency of 360 Hz, covering the P-wave, 
QRS-complex, and T-wave within one beat.

#### Feature Extraction and Classification

The 1D-CNN classifier, initially proposed by Nurmaini et al. (2020) for atrial fibrillation (AF) detection, was adapted 
in this study for the classification of abnormal–normal rhythm and beats. The model employed rectified linear unit (ReLU) 
functions, encompassing 13 convolution layers with varying filter sizes (64, 128, 256, and 512), along with five max 
pooling layers, and included two fully connected layers with 1,000 nodes each and one node for the output layer. Notably, 
the 1D-CNN required a three-dimensional input structure comprising n samples, n features, and timesteps.
