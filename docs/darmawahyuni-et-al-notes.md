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
