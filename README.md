# Bias Mitigation with CSIs
This repo contains the code for "Enhancing Speech Model Performance and Bias Mitigation via a Subgroup-Based Privacy Preserving Data Selection Strategy", submitted at InterSpeech 2024.

In this repository, you will find the code to replicate our experiments.  
We do not include the datasets used in the paper as they are publicly available: [FSC](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) and [ITALIC](https://huggingface.co/datasets/RiTA-nlp/ITALIC) for the Intent Classification (IC) task, and [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) for the Automatic Speech Recognition (ASR) task.


## Experimental Settings 

### Datasets 
We evaluate our approach on three publicly available datasets: Fluent Speech Commands (FSC) and ITALIC for the IC task in English and Italian, respectively, and LibriSpeech for ASR. 
FSC includes 30,043 English utterances, each labeled with three slots (action, object, location) that define the intent. 
ITALIC consists of 16,521 audio samples from Italian speakers, with the intent defined by action and scenario slots. 
For LibriSpeech, we utilize the *clean-360* partition, comprising 360 hours of clean audio samples.

### Metadata
For the above datasets we consider the following metadata when using DivExplorer for automatically extract subgroups: (i) demographic metadata describing the speaker (e.g., gender, age, language fluency level), (ii) factors related to speaking and recording conditions (e.g., duration of silences, number of words, speaking rate, and noise level), and (iii) intents represented as combinations of action, object, and location for FSC, or action and scenario for ITALIC.  
We discretize continuous metadata using frequency-based discretization into three distinct ranges, labeled as "low", "medium", and "high".   
Hence, continuous values are categorized into discrete bins based on their respective frequencies within the dataset. In the experiments, we explore all subgroups with a minimum frequency $s$ of $0.03$.

### Features employed to train the confidence models
We use the following features to train the confidence models:
- *Acoustic embeddings*: we use the embeddings extracted from the audio signal. Specifically, we use the [HuggingFace](https://huggingface.co/) implementation of the [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) and [whisper-base.en](https://huggingface.co/openai/whisper-base.en) models, and we extract the embeddings from the models' last hidden layer.
- *n-best list*: For LibriSpeech, we use the n-best list of the model, i.e., the list of the n most probable hypotheses for each utterance. 
- *Output probabilities*: for FSC and ITALIC we use the output probabilities of the model for each class.
- *Speech metadata*: we use the metadata extracted from the audio signal, including the number of words, number of pauses, speaking rate (word per second), etc.

### Models
We fine-tune the transformer-based wav2vec 2.0 base (ca. 90M parameters) and multilingual XLSR (ca. 300M parameters) models on the FSC and the ITALIC dataset respectively, and the whisper base (ca. 74M parameters) model on LibriSpeech. 
The pre-trained checkpoints of these models are obtained from the [Hugging Face hub](https://huggingface.co/models). 


### Hyperparameters and Hardware
IC task. We trained the models for $2800$ steps for FSC and $5100$ for ITALIC, with a batch size of 32, using the AdamW optimizer with a learning rate of $10^{-4}$, and $500$ warmup steps.   
ASR task. We trained the model for 3 epochs, with a batch size of 32, using the AdamW optimizer with a learning rate of $10^{-5}$.  
Experiments were run on a machine equipped with Intel Core $^{TM}$ i9-10980XE CPU, $2$ $\times$ Nvidia RTX A6000 GPU, $128$ GB of RAM running Ubuntu $22.04$ LTS. 

### Metrics 
We assess model performance using Intent Error Rate (IER) and F1 Macro scores for IC, and Word Error Rate (WER) for ASR. 
We also evaluate performance at the subgroup level, considering the IER and WER for the top-K challenging subgroups, with K in the range [2, 5].


### Baselines
We benchmark our approach against six baselines. 

*Random baseline*. We randomly add instances from the held-out dataset to the training data.

*KNN baseline*. We employ a K-Nearest Neighbors (KNN) classifier. We identify the K closest utterances from the training set for each instance in the held-out set, represented in the same input space as in our methodology. The selection of K is based on maximizing the performance, i.e., the identification of challenging subgroups, on the validation set.
We determine whether an utterance is challenging or not through majority voting among these neighbors. Predicted challenging instances are then included in the retraining process.

*Cluster-based baseline*. We use an unsupervised clustering approach to identify challenging subgroups inspired by the approach proposed in [Dheram et al., Interspeech, 2022](https://www.amazon.science/publications/toward-fairness-in-speech-recognition-discovery-and-mitigation-of-performance-disparities).
We first extract acoustic embeddings from audio samples and apply K-means to group them into similar clusters. 
In accordance with [Dheram et al., Interspeech 2022](https://www.amazon.science/publications/toward-fairness-in-speech-recognition-discovery-and-mitigation-of-performance-disparities)., we used 50 clusters as they are proven to adequately capture speech characteristics pertinent to ASR.
We then select clusters with the worst performance for data acquisition.

*CM-based baseline*. We use the CM to label the utterances. We include the samples predicted as erroneous by the CM in the training data. 

Finally, we employed two baselines that work as *oracles* since they assume the knowledge of the ground truth labels or the metadata, demographics included.

*Supervised oracle*. Similarly to what has been proposed in [Magar et al., 2023](https://www.sciencedirect.com/science/article/pii/S0927025623001611), we use an erroneous-sample-driven approach that incorporates instances predicted erroneously by the model into the augmented training data. 
This baseline assumes the prior knowledge of the ground truth labels for the tasks. 
This approach represents the oracle for the CM-based baseline.

*Metadata-based oracle*. We adopt the approach proposed in [Koudounas et al., ICASSP 2024](https://ieeexplore.ieee.org/document/10446326), which assumes access to metadata, demographic information included, for the samples in the held-out set that are to be acquired.
This approach represents the oracle for our proposal since, in our work, we use the CSI to predict the challenging subgroups without accessing metadata.

## License
This code is released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.