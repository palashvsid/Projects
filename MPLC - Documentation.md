PRODUCTION CATEGORIZATION USING MACHINE LEARNING

Powered By: Tredence Analytics Solutions

Introduction:

Product categorization for Ferguson, consists of classifying its current
inventory of ~3 million SKUs. Each of these SKUs needs to be classified
into a Product family to display and navigate on the client ecommerce
website. An efficient product categorization system will allow customers
to easily find the what they need. Currently only ~330K out of 3 million
SKU’s have been classified by a third-party contractor, at 7k SKU’s per
month. This is a manual and time consuming process with recurring
expenses. This accounts for ~95% of sales revenue for Ferguson.

To automate this process, a robust machine learning engine using
elements of supervised/ unsupervised learning has been developed. The
algorithm is developed based on the training data, i.e. existing SKU’s
which are already classified. The algorithm also accounts for the
ability to identify and handle the introduction of new product families
not available in the training data. Hence, this will allow Ferguson to
increase classification throughput, automate the process, minimize
expenses and eliminate manual errors.

Approach:

Identifying the features of products within a product category will
allow in efficient classification. Product classification tree is a
conceptual tool which will help in identifying such features and
automate the cataloguing process. The entire product classification
table can be imagined as a tree, with all products residing in the leaf
node. Features of the products within a leaf node will help determine
the feature vector for the node. Once the feature vector of a leaf node
is defined, features of unclassified products are extracted and matched
with feature vectors to determine which leaf node the product belongs
to. Features like product name, specification and attributes are used
for classification.

We followed a rigorous 4 step approach to identify such features and
classify the unclassified SKU’s,

1.  Data Cleaning & Data Preparation

    -   We identified the features / variables that influence the
        product classification

    -   Special characters and stop words were excluded from textual
        variables

    -   Lemmatization was carried out on all the textual descriptions

    -   Misclassified SKUs were identified manually and removed from the
        data analysis

2.  Data Modelling

    -   Standard text mining techniques were used to convert the data in
        to a document term matrix to build the model

    -   A set of machine learning algorithms including Ensemble methods
        were trained on a random sample

3.  Data Validation

    -   The trained models were tested on different set of SKUs

4.  Confidence Interval

    -   A cross validated threshold value was computed to signify the
        confidence of the predictions

An iterative process of classification is used, which uses already
classified SKUs as input for first iteration. Machine learning is an
iterative process that gets better over time with feedback loop. The
entire classification process

Data Cleaning and Data Preparation:

Once the required data is identified and collected, it goes through a
cleaning process to remove any discrepancies that might compromise the
result, following which a dataset with the required factors is prepared.
The prepared dataset is subsequently used for modeling. In this case
data is on ~70k SKU’s with product attributes for Pipe fittings and
Sinks was considered for the machine learning engine. The
features/variables that influence product classification were identified
as,

-   Product description

-   Vendor name

-   Linebuy description

-   Discount group

The products are classified as,

-   FOL Classification – Classification for website purposes.

-   GPH Classification – Classification for internal purposes.

*Callouts: *

-   Product Family & Product names are removed from the GPH Path to
    identify the classes

-   431 distinct FOL classes are present in the data

-   724 distinct GPH Path classes are present in the data

-   80% random samples from the dataset used to train the model and
    remaining 20% were used to test

*Data Exclusions:*

To avoid compromising the model the following discrepancies were
excluded from the data,

-   7k SKUs were excluded while training/validating the model since GPH
    was “unclassified”

-   18 FOL/ 122 GPH classes with single SKUs are excluded from training/
    validating

-   SKU with exact same attributes but tagged to different FOL/GPH were
    excluded from the analysis

    -   297 SKUs for FOL and 344 SKUs for GPH were excluded

After exclusions, the prepared dataset consisted of ~62k SKU’s which was
subsequently used for modeling.

To build the model the prepared dataset was converted in to a document
term matrix, through standard text mining techniques. A document-term
matrix or term-document matrix is a mathematical matrix that describes
the frequency of terms that occur in a collection of documents. In a
document-term matrix, rows correspond to documents (SKU’s in this case)
in the collection and columns correspond to terms (Dependent variables).

DATA MODELING:

The document term matrix of ~62k SKU’s is split into the training and
test set on an 80:20 ratio. The identified features are considered as
the independent variables and FOL link & GPH as dependent variables for
the modeling process,

The modeling process gives us the probability of a SKU to be classified
under a category. Hence, higher the probability, more the chances of the
SKU being under the corresponding product family. The prediction
accuracy is tested over multiple models and the model with the best
accuracy is selected for future predictions. The following models were
used for classification,

-   **Multinomial Naïve Bayes** - A Probabilistic classifier based on
    Bayes’ theorem

-   **Random Forest** – An ensemble learning method for classification/
    regression by constructing multiple decision tress

-   **Support Vector Machine** – A vector space algorithm for
    classification by identifying a hyperplane that differentiate
    classes

-   **Neural Networks** – A series of interconnected nodes that identify
    the relationships b/w dependent & independent variable

-   **Ensemble Method** – Learning algorithms that construct a set of
    classifiers and classify new data by voting

*Confidence Metric and Threshold Computation:*

The threshold is a probability based ratio to identify the confidence of
the model. The ratio of the 1<sup>st</sup> and 2<sup>nd</sup> highest
probabilities of a SKU was calculated to determine ratio-threshold. The
threshold is used to distinguish between “high-confidence” and
“low-confidence” predictions.

Coverage is the ratio of high confidence predictions to the total
predicted SKUs.

<img src="./media/image1.png" width="623" height="371" />

Cross Validated threshold value arrived at using the cutoff accuracy,67K
for FOL and 152K for GPH.

<img src="./media/image2.png" width="693" height="450" />

*Model Selection:*

The predictive ability of each model was compared to determine the
optimal model to be used for product classification.

A 5-Fold cross validation was carried out to validate the accuracy of
the model.

| **Model**                  | **Accuracy** |
|----------------------------|--------------|
| Multinomial Naïve Bayes    | 71.0%        |
| **Random Forest**          | **93.5%**    |
| **Support Vector Machine** | **94.5%**    |
| **Neural Network**         | **94.1%**    |

| **Model**                  | **Accuracy** |
|----------------------------|--------------|
| Multinomial Naïve Bayes    | 73.5%        |
| **Random Forest**          | **93.3%**    |
| **Support Vector Machine** | **94.7%**    |
| **Neural Network**         | **95.1%**    |

As seen, the neural network model gives a higher prediction accuracy in
both cases.

A cross validated threshold probability value was computed to indicate
the confidence of the model while scoring Unclassified SKUs. This
enables us to understand the high confidence predictions of each model,
i.e., the number of SKU’s classified by the model with 99% accuracy.

| Model                  | Accuracy | Coverage |
|------------------------|----------|----------|
| Random Forest          | 99%      | 19%      |
| Support Vector Machine | 99%      | 21%      |
| **Neural Network**     | **99%**  | **75%**  |

| Model                  | Accuracy | Coverage |
|------------------------|----------|----------|
| Random Forest          | 99%      | 19%      |
| Support Vector Machine | 99%      | 24%      |
| **Neural Network**     | **99%**  | **80%**  |

Hence, with classification of 75% SKU’s for FOL and 80% for GPH with an
accuracy of 99%, the neural network model was selected as the model for
future classifications.

Model Results:

Once trained on the ~50k SKU’s, the neural network model’s classifying
ability is verified through the test set. Product categorization is done
for FOL link and GPH classification. The required test set is fed into
the model, giving the following results,

*Predicted Results for FOL:*

The overall accuracy for FOL link classification is 94% with a standard
deviation of 1.1%.

Out of the 12,513 SKU’s in the test set, 75 % SKU’s are classified with
99% accuracy.

-   **Overall accuracy **

> **(9,273 + 2,487) / 12,513 = 94%**

-   **Accuracy with Threshold**

**9,273 / 9,367 = 99%**

-   **Coverage – 9,367 / 12,513 = 75%**

*Predicted results for GPH*:

The overall model accuracy for GPH is 95% with a standard deviation of
1%,

-   **Overall accuracy **

**(9,828 + 2,027) / 12,482 = 95%**

-   **Accuracy with Threshold**

**9,828 / 9,934 = 99%**

-   **Coverage – 9,934 / 12,482 = 80%**

As seen above the low confident predictions have a 79% accuracy for FOL
and GPH.

In the case of FOL; 38% of the wrong predictions, tags the immediately
next folder correctly while 72% tags the Parent folder correctly.

| **Actual**                                            | **Predicted**                                          |
|-------------------------------------------------------|--------------------------------------------------------|
| … &gt; Plastic Fittings and Flanges &gt; Plastic Wyes | … &gt; Plastic Fittings and Flanges &gt; Plastic Tees  |
| … &gt; Sinks &gt; Laundry Sinks                       | … &gt; Sinks &gt; Institutional Sinks &gt; Floor Sinks |

Example:

In the case of GPH; 36% of the wrong predictions, tags the immediately
next folder correctly while 68% tags the Parent folder correctly.

Example:

| **Actual**                                                            | **Predicted**                                                                              |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| …RESIDENTIAL SINKS| KITCHEN SINKS                                     | …RESIDENTIAL SINKS| BAR SINKS                                                              |
| …DRAINS, PARTS AND ACCESSORIES|COMMERCIAL FOUNDRY DRAINS| FLOOR SINKS | …DRAINS, PARTS AND ACCESSORIES | COMMERCIAL FOUNDRY DRAIN ACCESSORIES| GRATES OR STRAINERS |

*Key Outputs:*

The following results are obtained from the machine learning algorithm:

-   A 28X increase in the FOL throughput<sup>1</sup> with an overall
    accuracy of 94%

-   A 50X increase in the GPH throughput with an overall accuracy of 95%

| **Parameter**                                 | **Tredence Solution**                  
                                                                                         
                                                 **(FOL & GPH)**                         | **Current Solution** 
                                                                                                                
                                                                                          **FOL**               | **Current Solution** 
                                                                                                                                       
                                                                                                                 **GPH**               |
|-----------------------------------------------|----------------------------------------|----------------------|----------------------|
| Data Understanding + Processing + Preparation | ~2-3 weeks/category<sup>2</sup>        | NA                   | NA                   |
| Throughput                                    | ~10K SKUs / day                        | ~ 350 SKUs / day     | ~200 SKUs / day      |
| Monthly throughput<sup>3</sup>                | 10K \* 20 = **200K SKUs / month**      | ~7K SKUs / month     | ~4K SKUs / month     |
| Accuracy                                      | FOL - **94% ± 1.1%**                   
                                                                                         
                                                 GPH - **95% ± 1%**                      
                                                                                         
                                                 *After confidence cut-off “threshold”*  
                                                                                         
                                                 FOL – **99%** with **75%** coverage     
                                                                                         
                                                 GPH – **99%** with **80%** coverage     | 90%                  | 80%                  |

1 - Throughput computation for Tredence Solution is based on a 8 Core,
56 GB RAM server

2 - This is not a recurring effort for a given category. It is
recommended to review the process every 2-3 iterations to check if the
data has significantly changed

3 - Throughput includes training, validation & scoring

Data Validation:

The trained model was tested over different sets to check its accuracy.
We trained our models on different sets and made repeated predictions
for 500 samples of 500 observations each to find the variation in the
accuracy.

The FOL accuracy was found to be consistent over multiple random
samples,

| **FOL**           | **Average** | **Median** | **Std. Dev.** | **Maximum** | **Minimum** |
|-------------------|-------------|------------|---------------|-------------|-------------|
| **w/o threshold** | 94.1%       | 94.2%      | 1.1%          | 97.0%       | 90.6%       |
| **w/ threshold**  | 99.0%       | 99.0%      | 0.4%          | 100.0%      | 97.6%       |

The GPH accuracy was also found to be consistent over multiple random
samples,

| **GPH**           | **Average** | **Median** | **Std. Dev.** | **Maximum** | **Minimum** |
|-------------------|-------------|------------|---------------|-------------|-------------|
| **w/o threshold** | 95.1%       | 95.2%      | 1.0%          | 98.4%       | 92.2%       |
| **w/ threshold**  | 99.0%       | 99.0%      | 0.4%          | 99.8%       | 97.4%       |

Hence, we see that classification accuracy remains consistent for GPH
and FOL.

Increasing the threshold increases the accuracy and reduces the
variation but results in lesser data being predicted.

Prediction on Unclassified SKU’s:

Predictions on the 7K SKUs where GPH was not classified resulted in an
accuracy of 97% with a coverage of 57% for FOL.

Predicted results for FOL,

-   **Overall accuracy**

(3,986 + 1,547)/ 7,243 = 76.4%

-   **Accuracy with Threshold**

3,986 / 4,124 = 96.7%

-   **Coverage** – 4,124 / 7,243 = 57%

We trained our models on different sets and made repeated predictions
for 350 samples to find the variation in the accuracy. The Variation in
the accuracy for FOL predictions varies by 1.1% for a random sample of
350 SKUs.

| **FOL**          | **Average** | **Median** | **Std. Dev.** | **Maximum** | **Minimum** |
|------------------|-------------|------------|---------------|-------------|-------------|
| **w/ threshold** | 96.7%       | 96.8%      | 1.1%          | 99.53%      | 92.5%       |

The relatively low prediction accuracy is because, 71% of Carbon Steel
Flanges are misclassified as either Carbon Steel Weld Flanges or Carbon
Steel Forged Flanges.

-   50% of wrong predictions are found to be appearing from ‘Carbon
    Steel Flanges’

-   The ratio of SKUs of Carbon Steel Flanges to Carbon Steel Weld
    Flanges is low in the training set

-   Model assumes that the Unclassified SKUs are distributed across FOL
    classes like the training set distribution and hence predicts Carbon
    Steel Flanges as Carbon Steel Weld Flanges

| **FOL**                     | **\# SKUs Train set** | **\# SKUs (7K sample)** |
|-----------------------------|-----------------------|-------------------------|
| Carbon Steel Flanges        | 408                   | 1190                    |
| Carbon Steel Weld Flanges   | 1061                  | 25                      |
| Carbon Steel Forged Flanges | 362                   | 30                      |

***Misclassification Matrix for Carbon Steel Flanges***

| **FOL**              | **Carbon Steel Flanges** | **Carbon Steel Forged Flanges** | **Carbon Steel Weld Flanges** | **Grand Total** |
|----------------------|--------------------------|---------------------------------|-------------------------------|-----------------|
| Carbon Steel Flanges | 339                      | 240                             | 611                           | 1190            |
| **Percentage **      | **28%**                  | **20%**                         | **51%**                       |                 |

Challenges Faced:

*FOL: *

**1. Incomplete Data-** Our model can only predict what it knows

-   **Faucets**: Lack of data on “faucet” in product descriptions in our
    training set but the presence of “kitchen” or “lavatory” in the
    unclassified set confuses the current model

-   **PEX classifications**: Lack of data with “PEX” in product
    descriptions in our training set but presence of terms like “brass”
    and “couplings” confuses the current model

**2. Data from vendor sites**

-   Carbon Steel “Forged” vs “weld” O-lets/flanges

-   Check valves vs backflow preventers

*GPH:*

**1. Unextractable data: data in product names**

-   **Kitchen sinks vs bar sinks**: Product size determines the
    classifications but this information is present only where the class
    is a sink

-   Inconsistent patterns across classes and within the same class makes
    it difficult to extract

Scope for Improvement:

**Use entire dataset for model building**

-   Using entire dataset of the ~3 million SKUs wherever the GPH and FOL
    Link are present will improve estimations and coverage

-   Currently, we predict “Kitchen faucet” as “Kitchen sink” due to
    incomplete information

**Continual feedback iterations**

-   Using audited predictions to expand training set will improve future
    accuracies

-   Retrain models on expanded training set can happen on a regular
    monthly/quarterly basis to improve class coverage and accuracy

**Recommendations around single-SKU classes**

-   Recommendations can be made to reclassify current single-SKU classes
    into existing classes

-   Business rules can help to predict these correctly going forward

**Create post-prediction rules to accommodate class-specific nuances**

-   Using business “rules”, we can change predictions to accommodate
    nuances not present in data

-   “Kitchen sinks” vs “bar sinks”, “forged flanges” vs “weld flanges”


