# Introduction

Korean morphological analysis is the process of determining parts of
speech by identifying morphemes, which are the smallest units of
linguistic expression with independent meanings in a sentence. In an
isolating language, such as English, this identification can be achieved
relatively easily by tagging parts of speech sequentially. However, in
Korean, the nature of the agglutinative language requires separating
endings or postpositions and restoring inflections to their original
form. In addition, because the basic input of other Korean analysis
tasks is often a separate morpheme, the accuracy of the morphological
analysis significantly affects the performance of Korean analysis.
Modern high-performance deep learning methods in natural language
processing (NLP) use a tokenization process that breaks the text into
smaller units and converts each token into a vector as an input to the
computational model. Here, the token unit is mainly a subword unit, and
to reflect the characteristics of a Korean subword, tokenization with
separate morphemes is attempted in advance. Using the results of the
morphological analysis for this tokenization process improves the
overall performance of the analysis by reflecting the semantic units of
Korean. This requires a highly accurate and fast morphological analyzer.

Various approaches have been proposed for morphological analysis, which
is crucial in Korean language analysis. In general, when people
understand speech or written text, they attempt to make sense of it
using vocabulary and concepts that they are familiar with. While there
are approaches to use rules or dictionaries to reflect this
understanding, it becomes difficult to build and maintain a dictionary
for the vocabulary that appears in each text. Therefore, methods for
tagging syllable units without a dictionary have been proposed and
studies have been conducted to improve them . From a mechanical
perspective, syllable-by-syllable morphological analysis can be
performed either by tagging syllable-by-syllable and then applying a
base-form restoration dictionary, or by tagging syllable-by-syllable
with the base form already restored. However, syllable-by-syllable
morphological analysis has limitations, in that it is difficult to
accurately identify morpheme boundaries and learn long-term contextual
information as the length of the sequence increases. In this study, the
former is referred to as dictionary-based morphological analysis and the
latter as syllable-unit morphological analysis. Both methods are trained
on a manually labeled corpus and cannot accurately analyze new syllable
combinations or morphemes that do not appear in the training corpus.
Recently, with the development of the Internet and the spread of open
sources as well as open data, web texts, corpora, language resources,
and knowledge shared by different people have accumulated significantly.
The reduced cost of building and maintaining a dictionary provides a
significant opportunity to overcome the limitations of dictionary-based
methods.

Against this background, this study considers how the dictionary-based
morphological analysis method used by MeCab, an open software for
Korean and Japanese morphological analysis in tokenizers, which is an
essential preprocessing tool for deep learning, can be effectively
improved, and a method is proposed. The dictionary-based morphological
analysis method trained by the CRF (Conditional Random Fields) method
lists the candidate morphemes in the dictionary from a given sentence to
form a lattice structure connected by a directed graph and determines
the optimal morphological analysis path within it. The process of
determining the optimal path in the lattice uses the Viterbi algorithm,
which determines the path that minimizes the cost of each morpheme node
and the sum of the neighborhood costs of two consecutive morphemes. The
main types of errors in these dictionary-based morphological analysis
methods occur when new words that are not in the dictionary are used in
a sentence, or when the optimal path calculation selects the incorrect
result owing to bias. For example, it may be cost-effective to select
one long morpheme than several short morphemes, but this may often lead
to an incorrect analysis. The main motivation for this study was that
the path that minimizes the costs for the nodes and links may not be the
optimal path.

To identify cases in which a suboptimal solution is actually the best
solution according to the best path calculation, we modified the best
path calculation method to generate suboptimal analysis results and
verified the extent to which they are correct. Although there are
numerous different approaches to select the next-best path, we used the
method of replacing a morpheme node on the optimal path with a
lower-ranked node. As shown in
Table [tab:maximum-performance],
we confirm the extent to which the analysis performance can be improved
by replacing the optimal path with a lower-ranked node. We can consider
the problem of finding the correct answer among the generated
sub-optimals, similar to the problem of re-ranking search results in
information retrieval. In , the N-best analysis results generated by
the seq2seq model were re-ranked based on a convolutional neural network
to improve the performance. In this study, re-ranking was performed
using two BERT models of different types and forms, as proposed in .
Experimental results show that first-stage re-ranking improves the
performance by over 20% over existing written and spoken models, and
second-stage re-ranking with a different type of input and a different
type of pre-trained model further improves the performance by more than
30% over existing written and spoken models.

With this method, the performance of the dictionary-based morphological
analysis method could be further improved; however, the overall analysis
time increased when the morphological analysis system was configured,
including the re-ranking model itself. However, it is feasible to use
the results of multiple reranked morpheme analyses to update the
connection costs between morphemes in a dictionary, similar to the
backpropagation process in a typical neural network. It is also expected
that the morphological analysis system with improved connection costs
will be able to generate better re-ranking candidates, which will
further improve performance by doing so iteratively. Further research is
needed on this, and this study was limited to performance improvements
with two-stage reranking.

The main contributions of this study are as follows.

1.  **Further improvement of dictionary-based morphological analysis
    method using suboptimal analysis results**: We explore the
    possibility of performance improvement by introducing a method to
    replace the optimal path with a suboptimal node and propose a method
    to effectively improve the dictionary-based morphological analysis
    method through deep learning.

2.  **Extending the performance improvement by introducing a two-stage
    re-ranking model**: To improve the performance of dictionary-based
    analysis by re-ranking the morphological analysis results, we
    propose extending the performance improvement using different BERT
    models to perform two rounds of re-ranking.

3.  **A method for updating connection costs in the dictionary and
    suggestions for future research**: We propose a new method for
    updating dictionary connection costs based on re-ranked
    morphological analysis results. We also outline directions for
    future research, suggesting potential improvements.

These contributions provide important insights into the performance
improvement of Korean morphological analysis and the direction of future
research, and will serve as a useful reference for future researchers.

The remainder of this paper is organized as follows. In Section 2, we discuss
configuring and training a dictionary-based morphological analysis
system. In Section 3, we discuss the generation of
secondary results of morphological analysis, produce re-ranking data,
and propose a method for training a two-stage re-ranking model. In
Section  4, we discuss the results of the
performance improvement using the morphological analysis and re-ranking
models. In Section 5, we introduce previous research
cases related to this study. Finally, in Section 6, we conclude the study and discuss
its limitations and directions for future research.

# Morphological Analysis Model

## Korean Morphological Analysis Corpora

In this study, we used three major corpora to train and evaluate Korean
morphological analysis models, each of which has unique characteristics
and serves different research purposes.

**Sejong Corpus**: Originating from the 21st Century Sejong Project,
this corpus consists of a total of 15 million eojeols, including the raw
untagged corpus, and forms the backbone of Korean morphological analysis
research. It provides a wide variety of linguistic patterns and
structures that are important for baseline training and validation of
morphological analysis models, and has been used for performance
comparisons with other studies. In most of the previous studies, only a
part of the Sejong corpus was used for experiments, and in this study,
we used the dataset provided by the researchers in .

**UCorpus (University of Ulsan Corpus)**: This is an extension of the
Sejong corpus, which is constantly being maintained and added to by the
University of Ulsan, significantly increasing its volume to 63 million
eojeols and testing the adaptability and accuracy of the model to a
wider range of data. The expansion includes corrections to previously
raised errors  and a number of annotation outputs for new data,
providing a substantial basis for comprehensive linguistic analysis.

**Everyone’s Corpus**: Launched by the National Institute of the Korean
Language in 2020, the Everybody’s Corpus enriches the data landscape
with web texts and spoken language materials that reflect contemporary
language use. This modern corpus reflects the dynamic evolution of the
Korean language and is playing a pivotal role in improving models for
capturing the nuances of current Korean usage.

Table [tab:data-statistics] details
the specific number of sentences and words in each corpus, and the data
subsets extracted for model training and evaluation. During the training
data conversion process, we first removed duplicate sentences and
excluded sentences with annotation errors or other problems. We can see
that many duplicate sentences occur, especially in spoken language.

## Training Example Transformation

To train a dictionary-based morpheme analysis model effectively, the
morpheme-tagged corpus, typically represented in lemma form, must be
transformed to include the boundary information between morphemes in its
surface form. Crucial to this transformation is string alignment, a
process that accounts for the discrepancies between lemma forms and
surface forms in the Korean morphological analysis corpus.

In this study, string alignment was performed using the Smith-Waterman
algorithm, which uses a scoring matrix based on the similarity of the
grapheme unit of Korean letters for each word pair (as shown in
Figure [fig:sample]). Each aligned sentence
containing a morpheme tag was converted into a training sample tailored
for dictionary-based morphological analysis.

The resulting table in
Figure [fig:sample] illustrates this process.
Each row acts as a lexical unit. The first four columns contribute to
feature generation, whereas the last four columns facilitate
post-lemmatization. Using the morphological corpus above, a large number
of training samples can be generated according to the process shown in
Figure [fig:sample]. With the exception of the
evaluation samples, the remaining sentences were used to train the
dictionary-based morphological analysis model using the CRF algorithm.
The output of this training allowed the calculation of the costs
associated with each morpheme node and the linking of two consecutive
morphemes. This, in turn, allowed the discovery of an optimal path using
the Viterbi algorithm.

## Lattice Construction and Decoding

Figure [fig:lattice] presents a snapshot of
the lattice structure, which is an integral part of the morphological
analysis. (1) shows a fragment of the lattice structure formed when the
example sentence in
Figure [fig:sample] is entered. (2) shows the
optimal path determined using the Viterbi algorithm.

However, the path inferred by the trained model may differ from the
correct solution constructed by humans. The nodes marked with stars in
(1) represent correct nodes. The upper-left number of each node
indicates the ranking of the nodes accessible at each decoding point.
The choices made at certain moments deviate from the correct solution.
To improve analytical performance, mechanisms to correct these
discrepancies must developed.

# Re-ranking Model

## Motivation and Background

Although dictionary-based morphological analysis offers significant
advances, its optimal paths occasionally diverge from the correct
solutions that humans understand. This divergence underscores the need
for a model that reevaluates these primary results and reorients them to
achieve higher accuracy. This method, called re-ranking, involves
generating multiple analyses of an input and then reordering them based
on a new set of criteria or models, thereby improving the overall
quality of the results.

## Secondary Path Generation

Before reranking begins, multiple analyses, typically referred to as the
N-best paths, of the input sentence are generated. This involves
extracting the top N atoms from the lattice structure. In this study, a
novel approach is introduced to generate secondary paths, as shown in
(3) of Figure [fig:lattice], by selecting the
second-best node rather than each best node constituting the path from
the best-path result. Some of these secondary paths provided
alternatives that reconciled incorrect with correct answers. Similarly,
paths modified by favoring the third-best node were called tertiary
paths, and this naming convention was continued for subsequent paths. In
our preliminary test, the secondary paths, including the optimal and
suboptimal paths, were shown to cover the majority of the correct
morphological analyses as measured through human evaluations. (Refer to
Table [tab:maximum-performance]).

## BERT-based Re-ranking

Bidirectional Encoder Representations from Transformers (BERT) models
have revolutionized many natural language processing tasks by
understanding the context in which words appear in text. In this study,
we attempt to leverage the power of BERT to reorder the generated
secondary paths. We labeled the generated secondary paths with scores
related to the morphological analysis performance to fine-tune a
pre-trained BERT model specialized for Koreans with excess amount Korean
text. After pre-testing several scoring methods on a modest scale, we
found that using scores based on the degree of error rather than scores
based on accuracy by widen the gap between correct and incorrect answers
is effective for learning.

Once the BERT model is fine-tuned and trained for the re-ranking task,
it can predict a re-ranking score for each path in the secondary path
list. This implies that considering the context, morphological
organization, and other essential linguistic features of the path, the
model assigns a score to each path. The paths were then re-ranked
according to this score, and the path with the highest score was
selected for the best morphological analysis.

## Two-stage Re-ranking

Given the complexity of the Korean language, a single re-ranking step
does not constantly yield accurate results. Therefore, we propose a
two-step re-ranking approach as described in .

In the first step, we re-rank the secondary paths generated using the
BERT model, as described in Section 3.3. In the second
step, we introduced another BERT variant that was optimized for a
different set of linguistic features or trained on a different dataset.
This allowed us to perform a fine-grained re-evaluation, further refine
the list, and push more contextually accurate paths to the top.

As shown in Figure [fig:ranking], for a two-stage
reranking model, the first stage performs the first re-ranking, taking
as input a secondary path in morphologically tagged lemma form. It then
performs a second re-ranking, again taking as input the path re-ranked
in stage 1 and the original input sentence. This was conducted to
improve effectiveness, as it was ineffective given the same type of
input.

# Experimental Results

We evaluated the performance of the proposed deep learning-integrated
dictionary-based morphological analysis method. This section presents
the results of the experimental evaluation considering the improvements
over conventional methods and the effectiveness of our re-ranking model.

## Setup and Data

For our experiments, we used the Sejong corpus (versions used in ),
UCorpus, and Everyone’s Corpus. For comparison with previous studies,
the Sejong corpus was trained using a single model without separation.
The UCorpus and Everyone’s Corpus provided a separate spoken corpus with
drama scripts and broadcast dialogues, while UCorpus separated documents
that were considered to be close to spoken language and further
organized them into a semi-spoken corpus. Because UCorpus and Everyone’s
Corpus have synergistic effects when trained concurrently, we trained
the models separately for written and spoken language rather than
separating them by source. The statistics of the full data for the three
types of models are presented in
Table [tab:data-statistics]. Because
of the large volume of UCorpus, we randomly selected some of them to
train the actual model.

We transformed this organized morphological corpus using the
training-example transformation process described in Section 2.2 to
generate samples for training the dictionary-based morphological
analysis model.

## Evaluation Metrics

To measure the accuracy of the morphological analysis model, correctness
of the N-best path, and ranking accuracy of the reranking model, we used
the eojeol accuracy and morpheme F1 scores as evaluation metrics. To
verify that they produced the correct morphological analysis results, we
measured the degree of agreement with human annotations on the corpus.
However, owing to the slightly different criteria and styles of the
annotators who labeled the different types of corpora, including the
comparison with the MeCab-ko system, the following adjustments were
made:

-   Sentences containing unanalyzable tags (NF, NA, and NV) were
    excluded from both training and evaluation.

-   As for the tagsets, we excluded three unanalyzable tags from the 45
    Sejong tagsets and used 42 tagsets.

-   Each tag output by the MeCab-ko system was converted to the
    corresponding tag in the Sejong tagset.

-   Chinese characters were converted to Chinese character tags (SH)
    even if they were semantically used as nouns, and consecutive
    Chinese characters were converted to a single morpheme.

-   Similarly, symbol, numeral, ending, and postposition in the same tag
    were converted to a single morpheme, and decimal expressions were
    treated as a single morpheme, including the midpoint and the numbers
    before and after.

-   If the first lemma letter of the ending is ‘\[eo\]’, ‘\[yeo\]’, or
    ‘\[ah\]’, it is unified as ‘\[eo\]’, and if it is ‘\[eot\]’,
    ‘\[yeot\]’, or ‘\[ass\]’, it is unified as ‘\[eot\]’.

-   Root tags (XR) used alone without affixes were replaced with common
    nouns (NNG) because they are mainly used in the Sejong corpus only.

-   As mentioned in , connective endings (EC) and sentence-closing
    endings (EF) are not clearly defined in the tagging guidelines, and
    there are cases where they are used interchangeably in the corpus,
    to ensure that we evaluated them without distinguishing them.

-   The distinction between ‘\[geot\]’ and ‘\[geo\]’ is unclear in the
    tagging guidelines, and there are cases where they are used
    interchangeably in the corpus, hence, we did not distinguish between
    them.

-   Compound words can be interpreted as a single morpheme or as a
    combination of two or more morphemes or affixes, hence, we evaluated
    them without distinguishing between them.

-   Proper nouns can also be interpreted as common nouns depending on
    the point of view or perspective. Human annotators have slightly
    different standards, and thus they were also evaluated without
    distinguishing the nouns.

## Basic Performance

First, we compared the initial results of the dictionary-based
morphological analysis model trained using the method described in
Section 2 with those of
MeCab and syllable-based morphological analysis systems (refer to
Table [tab:performance-without-reranking]).
The results show that the dictionary-based method implemented is
superior to the existing MeCab system, but it differs from human
evaluations owing to the limitations mentioned above, and it does not
reach the performance of existing syllable-based morphological analysis
systems. We also found that the compatibility between the Sejong corpus
and other corpora is poor, as the model trained on the Sejong corpus has
guaranteed performance when evaluated on the Sejong corpus, and some
performance degradation occurs on other corpora.

## Re-ranking Performance

Upon integrating the BERT-based re-ranking model, we observed
substantial performance enhancement.
Table [tab:performance-with-reranking]
shows that the re-ranking model identified a better path in a
significant proportion of cases. The first-stage re-ranking exhibited a
performance improvement of over 20% compared with traditional models.
The subsequent re-ranking, leveraging a distinct type of input and a
different pre-trained model, further augmented the performance by more
than 30%.

Next, we performed the BERT-based re-ranking described in Section 3 and compared its
performances. Three pre-trained language models, KPF-BERT, ETRI-ELECTRA,
and ETRI-RoBERTa, which are known to perform well in other Korean
language understanding tasks, were used to fine-tune the re-ranking
model. KPF-BERT received Korean sentences as input, ETRI-ELECTRA
received morphologically tagged sentences as input, and ETRI-RoBERTa
receives morpheme-separated sentences as input. For KPF-BERT and
ETRI-RoBERTa, there may be problems with model learning because of the
separation of morpheme tags into letter units during the tokenization
process when the morpheme analysis results were received as input and
re-ranked, hence, the morpheme tags of the mid-level classification unit
were added as new tokens, and then training was performed. As shown in
Figure [fig:ranking], for these three types of
pre-trained language models, we first performed training with a
re-ranking model using only the morphological analysis results and then
performed training with a second re-ranking model using the top five
morphological analysis results from the first re-ranking results along
with the input sentences for other types of pre-trained language models.
Preliminary tests indicate that using the same type of model or input
results in a nearly identical retrained model, with no further
improvement in performance.

The sentences used for training the re-ranking model were selected from
those used for training the dictionary-based morphological analysis
model: 190,000 sentences from the Sejong corpus, 240,000 sentences from
the written language of the combined UCorpus and Everyone’s corpus, and
360,000 sentences from the spoken language of the combined UCorpus and
Everyone’s corpus. For a large number of sentences, we adopted a
floating-point 16-bit technique while using four GPUs for distributed
training and significantly reduced the time required for training. In
addition, the minibatch size was 120 with a maximum sequence length of
384 because the first re-ranking uses only the morphological analysis
results as input. The minibatch size was 40 with a maximum sequence
length of 512 because the original input sentences were given as input
along with the first morphological analysis results. The learning rate
was set to 2 × 10<sup>−5</sup> and AdamW is used as the optimization
algorithm.

Table [tab:performance-with-reranking]
shows that incorporating the re-ranking model significantly improves the
performance compared with no re-ranking. The error reduction rate (ERR)
of the performance change from the existing model on eojeol accuracy is
shown as 29%, 27%, and 20% for the Sejong corpus, combined written
corpus, and combined spoken corpus, respectively, with the first round
of re-ranking, and the second round of re-ranking improved the
performance by increasing the rate to 34%, 34%, and 32%, respectively.
These performance improvements demonstrate that the dictionary-based
morphological analysis model outperforms traditional syllable-based
morphological analysis systems, including numerous pre- and
post-processing rules and dictionaries.

## Comparison to Other Studies

We found that the proposed transformer-based re-ranking technique
consistently improved the results of the existing morphological analysis
models. These results confirm that it opens up new possibilities by
further improving the results of existing traditional machine-learning
models in the field of Korean morphological analysis. Finally, because
the major related studies proposed in the literature were mostly
conducted on the Sejong corpus, we compared the performance improvement
of the Sejong corpus with the results of previous studies. Although it
is difficult to make a direct comparison because of slight differences
in implementation conditions and evaluation criteria. The proposed
dictionary-based morphological analysis model is not up to the latest
research results; however, by incorporating a re-ranking model, it can
secure a performance that is comparable to existing research.

The entire morphological analysis model, including the re-ranking model,
is not suitable for real-time processing. However, it is expected that
by reflecting the cases whose ranks are changed through the re-ranking
model as feedback to the dictionary-based morphological analysis model,
it will be feasible to obtain near-improved morphological analysis
performance. The improved dictionary-based morphological analysis model
can then be used as input to the re-ranking model; therefore, it is
expected that a gradually improvement in morphological analysis model
can be obtained through this iterative feedback loop.

# Related Work

Korean morphological analyses have seen an influx of various
methodologies in recent years. The nature of the Korean language, being
agglutinative, introduces challenges that have propelled researchers to
develop inventive solutions, many of which have laid the groundwork for
future research.
Table 1
provides a concise comparison of the methodologies and key concepts of
key studies that are directly or indirectly related to this research,
giving a brief overview of the different methods of morphological
analysis.

## Traditional Dictionary-based Approaches

The earliest attempts at Korean morphological analysis relied heavily on
rule- and dictionary-based methods. These methodologies employ
predefined sets of linguistic rules or large dictionaries to detect
morphemes and determine parts of speech. One of the most significant
advantages of this approach is its deterministic nature, which can lead
to high accuracy when the input text closely adheres to the rules or
dictionaries used. However, it is challenging to scale and update them,
particularly with the constant evolution of language and the emergence
of new words. The dynamic nature of language, particularly in the
Internet age, has made the maintenance of comprehensive dictionaries
labor-intensive.

## Syllable-unit Morphological Analysis

To overcome the limitations of dictionary dependence,
syllable-by-syllable morphological analysis has emerged as an
alternative. This method either tags each syllable and then applies a
base-form restoration dictionary or tags the syllable with the base
form already restored. One notable drawback is the challenge of
accurately pinpointing morpheme boundaries. Furthermore, as the
sequences increase in length, the system finds it increasingly
challenging to understand long-term contextual data.

## Recent Deep Learning Approaches

The incorporation of deep learning into Korean morphological analysis
has brought significant advancements to the field. Existing deep
learning methods typically employ architectures like Bidirectional Long
Short-Term Memory (Bi-LSTM) networks, Convolutional Neural Networks
(CNNs), and Transformer-based models. These approaches focus on
understanding language context and sequence, utilizing the ability of
these models to capture long-range dependencies and intricate patterns
in text data. For example, Bi-LSTM-CRF models, extensively used for
sequence labeling in morphological analysis, leverage LSTM’s capacity to
remember long-term dependencies and CRF’s proficiency in sequence
prediction.

In contrast, our method innovatively integrates the re-ranking concept
with BERT-based models for Korean morphological analysis. Unlike
traditional deep learning methods that primarily use
sequence-to-sequence or sequence labeling approaches, our method
generates suboptimal paths using dictionary-based techniques, which are
then re-ranked by BERT models. This dual approach, leveraging BERT’s
contextual understanding, allows for a more detailed and accurate
morphological analysis. The distinction of our approach lies in its
ability to address the complexities of the Korean language. By
generating and re-ranking suboptimal paths, our method can identify and
rectify anomalies that standard deep learning models may miss. This
innovative strategy combines the precision of dictionary-based methods
with the contextual comprehension of BERT models, marking a significant
advancement in the field, especially for languages with intricate
morphological structures like Korean.

## Integrating Dictionary-based and Deep Learning Approaches

Tokenisation, a fundamental process in NLP deep learning models,
involves breaking down text into smaller units and converting these
tokens into vectors for computational processing. For Korean, with its
complex morphological characteristics, tokenisation that respects
morpheme boundaries is crucial. This is because it can not only
accurately capture the linguistic nuances of Korean, but also improve
the overall performance of deep learning models. This is especially
important given the agglutinative nature of Korean, where words are
formed by combining morphemes with different semantic and syntactic
information.

In this respect, the combination of dictionary-based morphological
analysis methods and deep learning approaches used by MeCab, a fast and
lightweight morphological analyser used for tokenisation of Korean and
Japanese, is quite useful. The dictionary-based morphological analysis
used here uses a model trained with CRFs to form a lattice structure as
in  to identify the optimal path for morphological analysis. While this
method provides a degree of accuracy and speed, it does not achieve the
high accuracy achieved by modern deep learning.

Our research sought to bridge this gap by effectively combining these
dictionary-based morphological analysis methods with the contextual
understanding capabilities of deep learning. Future research should
continue to refine these hybrid methods to explore the possibility of
end-to-end models that seamlessly combine the strengths of traditional
dictionary-based analysis with the adaptive capabilities of deep
learning. This direction promises significant advances in morphological
analysis and will further push the boundaries of Korean language
processing.

# Conclusion

This study signifies a progressive stride in Korean morphological
analysis by seamlessly merging conventional dictionary-based techniques
with advanced deep learning methodologies. Our findings indicate that
while relying solely on dictionary-based morphological analysis does not
surpass the efficacy of some existing models, the integration of a
BERT-based re-ranking system notably enhances accuracy, establishing a
new standard in this domain.

While the performance improvement increases computational demand, the
introduced methodology provides a promising avenue for continuous
enhancement. This innovative amalgamation of classical dictionary
approaches and cutting-edge machine-learning methodologies paves the way
for groundbreaking advancements in the intricate and multifaceted
domains of Korean linguistic processing.

Future endeavors in this domain should emphasize the refinement of this
harmonious integration to achieve even higher precision in morphological
analysis while optimizing computational efficiency. Moreover, our
observations indicate the potential for employing a probabilistic model
to discern areas where inaccuracies are likely to arise, thus enabling
the retrieval of more accurate interpretations from a narrower candidate
pool. The parallels between this initiative and the challenges of
translation quality estimation suggest that insights from the latter can
bolster the efficacy of our approach.
