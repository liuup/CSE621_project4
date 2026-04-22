# **CSE621 Project 4: Large Language Models for Classification and Clustering** 

**Resources:**  
**Scaffolding Python notebooks**:  
[Colab notebook for Encoder](https://colab.research.google.com/drive/1cZP_qk3Eb04sQXmYHSe2tq-mp2i09Uxg?usp=sharing)  
[Colab notebook for Decoder](https://colab.research.google.com/drive/1GBYZFAC46R-B8o0Wx2DFi2kslmnWfB2R?usp=sharing)

**Videos** of implementation demos (see under Resources for the LLM unit for additional lectures and demos)

* [Colab notebook for Encoder](https://colab.research.google.com/drive/1cZP_qk3Eb04sQXmYHSe2tq-mp2i09Uxg?usp=sharing)  
* [Colab notebook for Decoder](https://colab.research.google.com/drive/1GBYZFAC46R-B8o0Wx2DFi2kslmnWfB2R?usp=sharing)  
*   
* Project 4 [Walk-through-code demo recording](https://drive.google.com/file/d/1dy2AV7NZBO2KaphgIJOTmzQasPXRLTmv/view?usp=sharing) on **Encoder-only LLM** (Project 4 notebook)  
* Project 4 [Walk-through-code demo recording](https://drive.google.com/file/d/17GkLToIm1bYK0qu4Qpb4XJ2n0mUVjTMc/view?usp=drive_link) on **Decoder-only LLM** (Project 4 notebook

   
**Objective**:  
The goal of this project is to provide practical experience working with some common LLM workflows using the scaffolding code provided above. 

For all of the following tasks, please use the **BBC News Dataset** which can be found [here](https://huggingface.co/datasets/SetFit/bbc-news). This is a small dataset that is a nearly identical task to the 20newsgroups dataset (shown in the demo notebooks), but should be simpler and faster to work with. This dataset comes **pre-split**, so **you should evaluate all of your models on the default test split**.

Also, if you have not already done so, we recommend [signing up for the free education tier for Colab](https://colab.research.google.com/signup) (to use better accelerators, which are extremely recommended for this project). **If you use the T4 GPU, you should have no issue completing all of the steps in the project in time** using the scaffolding notebooks provided above.

# Part 1\. Classification

## Part 1.1 Encoder

1. Use a pre-trained Encoder model as a frozen layer, **train only a classifier head**.

   1. Don’t worry too much about finding perfect training parameter configurations for this. Maybe try a few options for learning rate or use the default in the notebook. Just report the best performance you find.

2. Use a pre-trained Encoder model with no training, in a **zero-shot** fashion.

## Part 1.2 Decoder

*For this section, we recommend using a small instruction-tuned model, such as Qwen2.5-1.5B-Instruct, for the best results. As an example, you can look [here](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?official=true&params=0%2C3) for other options.*

1. Following the guide in the provided scaffolding notebook above, use an autoregressive decoder-only model to perform **zero-shot classification** on your **test** dataset.

2. Experiment with **few-shot classification** by modifying your instruction prompt to include example/label pairs derived from the training set. The simplest way to do this is to just randomly sample an example from the different classes (*you can also optionally do more interesting/complex things, such as using clustering to derive medoid examples that are representative of different regions of space*)

## Part 1.3 Classical

1. Train a classical model on the text, i.e. Naive Bayes, Logistic Regression, SVM, etc. to compare against your language models. **You** **can** **reuse part of Project 1** here. 

## Putting all the Above Together: What to Compare in Part 1

At a minimum, you should have at least 5 classifiers to compare: 

1. **2 encoder-based classifiers from Part 1.1,**   
2. **2 decoder-based classifiers from Part 1.2, and**   
3. **1 classical model from Part 1.3.** 

Use traditional classification **evaluation** metrics of **Accuracy, Precision, Recall, and F-Score** to evaluate and compare performance. Remember to **evaluate all of your models on the default test split as provided.** Additionally, make note of the **runtimes** for each approach.

# Part 2\. Clustering

## Part 2.1 Encoder

Use a pretrained Encoder model to obtain document **embeddings**. Do this **twice**, one time **using the \[CLS\] pooled token**, and one time taking the **average** of **all non-special tokens**.

## Part 2.2 Classical

Choose a clustering algorithm from Project 2 and perform clustering on the embeddings. First evaluate using external labels (using external validity measure). Then compare this result against an alternative classical representation of the text of your choosing (bag of words: BoW, TF-IDF, LSI, etc.). Compare and show your results.

## Putting all the Above Together: What to Compare in Part 2

For this part of the project, you should have a **total of 3 results to compare using similar cluster validity metrics and cluster visualization methods as Project 2**: 

1. 2 different clusterings on language model-derived **embeddings** **from Part 2.1**  
   1. **using the \[CLS\] pooled token**, and 

   2. taking the **average** of **all non-special tokens**.

2. 1 clustering on **classical** representations **from Part 2.2.**

**What to return:**

* Your notebook/code  
* A **brief** report (ideally \~3-5 pages **max**) containing the following subsections under separate Heading sections for **Part 1 and for Part 2**:  
* For **each Part**, include the following:  
  * **Methodology:**  
    * Explain what you did, briefly describe the methods used and if you made any alterations.

  * **Experimental Results & Analysis:**  
    * Provide **plots** to summarize your validity metrics and runtimes.  
    * At minimum you should have a table and a plot that succinctly compare the performance of all classifiers in Part 1\.  
    * Provide a table/plot that compares the clustering performance of all the different approaches in Part 2\.   
    * Provide an **analysis** of your results. What worked best? Why do you think it worked best? Just give some brief thoughts about the things you’ve explored in this notebook.

  * **Conclusion:**  
    * A short summary again of what you have done, what your key findings were, and how you felt about this project in terms of what you learned (e.g. did you feel that you learned how to apply transformer-based language models to tasks like classification and clustering?)  
* **Additionally, record a brief \~5 minute video sharing your screen and following the provided structure:**  
* \- Demo your notebook/code step-by-step, detailing what methods were used (\~3 minutes)  
* \- Summarize/discuss your findings, things you thought were interesting, etc. (\~1 minute)  
* \- Specify the most challenging/rewarding aspects of the project, and why (\~1 minute)  
* 

**It is preferred if you can provide a link to your recording rather than the file itself. A couple of examples you can try:**

* \- MS Teams and Microsoft Sharepoint (please make the recording visible through the link you submit)  
* \- Google Drive (upload Mp4 to drive \--\> share file by link)  
* \- YouTube (unlisted video with link)

# **Rubric:**

| Classification |  |
| ----- | :---- |
| **Encoder fine-tuning** | **7.5** |
| **Encoder zero-shot** | **7.5** |
| **Decoder zero-shot** | **7.5** |
| **Decoder few-shot** | **7.5** |
| **Classical model** | **7.5** |
| **Proper procedure** | **10** |
| **Clustering** |  |
| **\[CLS\] embeddings** | **5** |
| **Mean pool embeddings** | **5** |
| **Classical representation of text** | **5** |
| **Proper procedure** | **10** |
| **Report** |  |
| **Methodology** | **5**  |
| **Table for classification results** | **5** |
| **Plot(s) for classification results** | **5** |
| **Table/plot for clustering results** | **5** |
| **Analysis & discussion of results** | **5** |
| **Conclusion** | **5** |
|  | **100 (minus 5 points for missing video)** |

