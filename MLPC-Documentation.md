<h1> PRODUCT CATEGORIZATION USING MACHINE LEARNING </h1>

<h3> Powered By: Tredence Analytics Solutions </h3> 

<h2> Introduction</h2>
The client requires an efficient product classification system, for easy navigation and product display on the client e-commerce website. This will allow customers to find their exact needs. Currently, out of the 3 million SKUs in its inventory only 330k SKUs have been classified by a third-party contractor, at 7k SKUs per month. This is a manual and time consuming process with recurring expenses.
<br>
Hence, we are building a robust machine learning engine using elements of supervised/ unsupervised learning. The algorithm is developed based on existing SKUs which are already classified. The algorithm also accounts for the ability to identify and handle the introduction of new product families not available in the training data. Hence, this will allow the client to increase its classification throughput, automate the process, minimize expenses and eliminate manual errors.

<h2> Data Sources </h2>
Data files on SKU attributes and abbreviation mapping file were extracted from STEP, Ferguson's master data management system. The data was pulled, understood and the relevant subset was derived with the help of three teams:

1. IT MDM team (Point of contact: Richard Barber)

2. E-business team (Point of contact: Jaclyn Aulich/ Kelly Amavisca)

3. Product services team (Point of contact: Brittany Merritt)

<h3>Sample Files</h3>
**Sample.csv:** A sample subset of the 70k SKUs belonging to the categories of Sinks and Pipe fittings
**abbrevation_mapping_dict.csv:** The SKU data uses short forms for its product descriptions. Hence, for proper classification, this field is expanded using the abbreviation mapping file.


<h2> Approach </h2> 
Identifying the features of products within a product category will allow in efficient classification. We trained the machine learning
algorithm on existing classified data and predicted the product family of unclassified products to test the accuracy of the model. To achieve this, a rigorous 4 step approach was followed,

1.  **Data Cleaning & Data Preparation**

    -   We identified the features / variables that influence the product classification

    -   Special characters and stop words were excluded from textual variables

    -   Lemmatization was carried out on all the textual descriptions

    -   Misclassified SKUs were identified manually and removed from the data analysis

2.  **Data Modeling**

    -   Standard text mining techniques were used to convert the data into a document term matrix to build the model

    -   A set of machine learning algorithms including Ensemble methods were trained on a random sample

3.  **Confidence Metric and Threshold Computation**

    -   A cross validated threshold value was computed to signify the confidence of the predictions

4.  **Model Testing**

    -   The trained models were tested on different set of SKUs


<h3>Data Cleaning and Data Preparation</h3> 

The data considered for the algorithm consists of ~70K SKU attributes. The data goes through a cleaning process to remove any discrepancies that might compromise the result. The prepared dataset is subsequently used for modeling.

**Data Exclusions:**

To avoid compromising the model, the following discrepancies were excluded from the data:

-   7k SKUs were excluded while training/validating the model since GPH was “unclassified”

-   18 FOL/ 122 GPH classes with single SKUs are excluded from training/validating

-   SKU with same attributes, but tagged to different FOL/GPH were excluded from the analysis

-   297 SKUs for FOL and 344 SKUs for GPH were excluded

After exclusions, the dataset was reduced to ~62K SKUs

Independent Variables:

-   Product description

-   Vendor name

-   Linebuy description

-   Discount group

Dependent Variables:

-   FOL Classification

-   GPH Classification

Example: Independent variables,

| **&lt;ID&gt;** | **PROD\_LONG\_DESC**                 | **Primary Vendor** | **DISC\_GROUP\_ID\_DESC**   | **LINEBUY\_ID\_DESC** |
|----------------|--------------------------------------|--------------------|-----------------------------|-----------------------|
| Prod-1021748   | 33X19 4H 2B SS SR KITC SINK \*STYLIS | JUST MANUFACTURING | JUST SS COMM WALL HUNG SINK | JUST MFG              |

Example: Dependent variables,

| **&lt;ID&gt;** | **GPH Full Path**                                                                                                                                                                                                                                                                                                            | **&lt;FOL Link&gt;** |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| Prod-1021748   | 16 PLUMBING - FINISHED>>04 PLUMBING FIXTURES, PARTS AND ACCESSORIES>>19 COMMERCIAL SINKS>>08 KITCHEN SINKS>>4-Hole 2-Bowl Drop-In, Self-Rimming and Topmount Rectangular Kitchen Sink with Center Drain>>08 JDLADA1933A46DCR 33X19 4H 2B SS SR KITC SINK \*STYLIS | Kitchen Sinks        |

The prepared dataset is subsequently converted to a document term matrix, through standard text mining techniques. A document-term matrix or term-document matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents. In a document-term matrix, rows correspond to documents (SKUs in this case) in the collection and columns correspond to terms (Dependent variables).

<h3>Data Modeling</h3>
The document term matrix is split into the training and test set on an 80:20 ratio. The identified features are considered as the independent variables and FOL link & GPH as the dependent variables for the modeling process,

1.  Random sample (80% sample)

2.  Data cleaning and feature engineering

3.  Machine learning – The following model were used for classification

    -   Multinomial Naïve Bayes - A Probabilistic classifier based on Bayes’ theorem

    -   Random Forest – An ensemble learning method for classification/ regression by constructing multiple decision tress

    -   Support Vector Machine – A vector space algorithm for classification by identifying a hyperplane that differentiate classes

    -   Neural Networks – A series of interconnected nodes that identify the relationships b/w dependent & independent variable

    -   Ensemble Method – Learning algorithms that construct a set of classifiers and classify new data by voting

4.  Validation and fine tuning (20% Sample)

<h3>Confidence Metric and Threshold Computation </h3>
Threshold is a probability based ratio to identify the confidence of the model. The ratio of the 1<sup>st</sup> and 2<sup>nd</sup> highest probabilities of a SKU is calculated to determine the threshold. It is used to distinguish between “high-confidence” and “low-confidence” predictions.

<h3> Model Testing</h3> 
The predictive ability of each model was compared to determine which is to be used for product classification. A 5-Fold cross validation was carried out to validate the accuracy of the model.

The neural network model is selected due to its higher accuracy and coverage.

Once trained on the ~50k SKUs, the neural network model’s classifying ability is verified on the test set. Product categorization is done for FOL link and GPH classification. The required test set is fed into the model, giving the following results,

<h4>Predicted Results for FOL</h4>
The overall accuracy for FOL link classification is 94% with a standard deviation of 1.1%.

| **Total SKUs 12,513** | **High Confidence 9,367 (75%)** | **Low Confidence 3146 (25%)** |
|------------------------|---------------------------------|-------------------------------|
| Correct predictions    | 9,273 (99%)                     | 2,487 (79%)                   |
| Wrong predictions      | 94 (1%)                         | 659 (21%)                     |

**Overall accuracy: (9,273 + 2,487) / 12,513 = 94%**

**Accuracy with Threshold: 9,273 / 9,367 = 99%**

**Coverage: 9,367 / 12,513 = 75%**

<h4>Predicted results for GPH</h4>

| **Total SKUs 12,482** | **High Confidence 9,934 (80%)** | **Low Confidence 2,548 (20%)** |
|------------------------|---------------------------------|--------------------------------|
| Correct predictions    | 9,828 (99%)                     | 22,027 (79%)                   |
| Wrong predictions      | 106(1%)                         | 521 (21%)                      |

**Overall accuracy: (9,828 + 2,027) / 12,482 = 95%**

**Accuracy with Threshold: 9,828 / 9,934 = 99%**

**Coverage: 9,934 / 12,482 = 80%**


The trained model was tested over different sets to check its accuracy. We trained our models on different sets and made repeated predictions for 500 samples of 500 observations each to find the variation in the accuracy.

The FOL accuracy was found to be consistent over multiple random samples

| **FOL**           | **Average** | **Median** | **Std. Dev.** | **Maximum** | **Minimum** |
|-------------------|-------------|------------|---------------|-------------|-------------|
| **w/o threshold** | 94.1%       | 94.2%      | 1.1%          | 97.0%       | 90.6%       |
| **w/ threshold**  | 99.0%       | 99.0%      | 0.4%          | 100.0%      | 97.6%       |

The GPH accuracy was also found to be consistent over multiple random samples,

| **GPH**           | **Average** | **Median** | **Std. Dev.** | **Maximum** | **Minimum** |
|-------------------|-------------|------------|---------------|-------------|-------------|
| **w/o threshold** | 95.1%       | 95.2%      | 1.0%          | 98.4%       | 92.2%       |
| **w/ threshold**  | 99.0%       | 99.0%      | 0.4%          | 99.8%       | 97.4%       |

Increasing the threshold increases the accuracy and reduces the variation but results in lesser data being predicted.

<h4>Prediction of Unclassified SKUs</h4>
Predictions on the 7K SKUs where GPH was not classified resulted in an accuracy of 97% with a coverage of 57% for FOL.

Predicted results for FOL,

| **Total SKUs 7,243** | **Confident Predictions 4,124(56%)** | **Unclassified SKUs 3,119(44%)** |
|-----------------------|--------------------------------------|-----------------------------------|
| Correct predictions   | 3,986(97%)                           | 1,547(49%)                        |
| Wrong predictions     | 138(1%)                              | 1,572(51%)                        |

**Overall accuracy**: (3,986 + 1,547)/ 7,243 = 76.4%

**Accuracy with Threshold**: 3,986 / 4,124 = 96.7%

**Coverage**: 4,124 / 7,243 = 57%

<h2>Impact</h2>
The machine learning algorithm enables:

-   A 28X increase in the FOL throughput with an overall accuracy of 94%

-   A 50X increase in the GPH throughput with an overall accuracy of 95%

<h2>Scope for Improvement</h2>
-   Use entire dataset for model building

-   Continual feedback iterations

-   Recommendations around single-SKU classes

-   Create post-prediction rules to accommodate class-specific nuances


<h2>Code- Working Flow</h2>
The code working flow is as follows,

1.  First, please set the working directory where the provided sample and abbreviation data is stored. Then hit run.

2.  **User defined functions:** These are the functions created for the subsequent data cleaning, preparation and modeling purposes.

    1.  Remove\_last\_two\_pipes: Function to remove the last two levels in the GPH path. This is to derive the product family of
        that product.

    2.  Clean\_text: Function to clean the data. It removes special characters, so that the result is not compromised.

    3.  Replace\_space: Function to replace the spaces in columns with “\_”.

    4.  Map\_abbreviation: Function to derive expand all abbreviations in the data, i.e. derive the complete product description.

    5.  Remove\_duplicate\_data: Function to exclude duplicate SKUs w.r.t to the independent and dependent variable attributes.
        These SKUs are misclassified and must be removed.

    6.  Create\_tdm\_dataframe: Function to create the required document term matrix. The independent variables are fed in to the
        function to derive its unigrams and bigrams as the columns.

    7.  Ratio\_threshold: Function to calculate the threshold value. This is used to mark “high confidence” and “Low confidence” predictions.

3.  **Data Preparation:** The necessary information is derived as “data” and is subsequently prepared as,

    1.  The columns of the dataset are cleaned using the clean\_text and replace\_space functions.

    2.  The GPH column of the dataset is cleaned using the remove\_last\_two\_pipes function.

    3.  The product long description abbreviations are expanded using the map\_abbreviation function.

    4.  All null and blank fields in the product long description field is replaced with the product description field (A shorter
        version of product long description).

    5.  All SKUs with GPH tagged as onboarding are replaced. These SKUs are unclassified and hence cannot be used to train the
        machine learning engine.

    6.  The final prepared dataset for machine learning is named as “final\_data”

4.  **Product classification for GPH:** This step involves training the model to predict GPH classifications

    1.  **Document term matrix:** It describes the frequency of terms that occur in a collection of documents.

        1.  The list of independent variables and dependent variables are set.

        2.  Misclassified data is removed from “final\_data” using remove\_duplicate\_data function. The resultant dataset is assigned               as “gph\_final\_data”.

        3.  The document term matrix is prepared from “gph\_final\_data” using the create\_dtm\_dataframe function. Each independent                 variable is fed into the function, creating unigrams and bigrams for “Product long description” and unigrams for “Linebuy id             desc” & “Disc group id desc”. The vendor names are assigned dummy variables (1 or 0).

        4.  “Dtm\_data1”,” Dtm\_data2”, “Dtm\_data4” and vendor\_dummies” are”compiled together to derive the document term matrix. The             actual GPH classes are also attached to this matrix.

        5.  In case of a column named “fit” it is renamed to fit\_feature to avoid any discrepancy with the compiler.

        6.  All classes with a single SKU are excluded from the dataset.

    2.  **Neural network model for GPH:**

        1.  The model metrics are set and the dataset is split into the training and test set (80:20 ratio).

        2.  The model is trained and stored in “nn\_predicted\_gph”. Subsequently the predicted classes and predicted probabilities are             stored in ‘’predicted\_nn\_gph” and “prob\_nn\_gph” reapectively.

        3.  The accuracy of the model is also derived and stored in “gph\_accuracy”.

    3.  **Threshold value calculation:** It is calculated using the ratio\_threshold function.

5.  **Product Classification for FOL link:** This step involves training the model to predict FOL classifications. It follows the same
    process as **STEP 4**, but here the dependent variable is replaced as “**FOL Link**”.

6.  **Product classification on the unclassified SKUs:**

    1.  **Prediction for FOL Link**

        1.  Here the dataset is derived as final\_data1. This consists of all SKUs with GPH tagged as onboarding. We predicted the
            result on the FOL link which was available in the data.

        2.  The required independent and dependent variables are selected.

        3.  Misclassified SKUs are removed from the data using remove\_duplicate\_data function.

        4.  The document term matrix is prepared using the create\_dtm\_dataframe function & vendor\_dummies. Then they are compiled                 into a single dataframe.

        5.  From line 446 to 469, the document term matrix created is modified so that it has the same columns as the training dataset               on which the model learnt. Here, columns not present in the document term matrix is attached and tagged as zero.
            This is to ensure the machine learning engine scores based on what it was trained, thereby giving proper classifications.

        6.  The independent variables and dependent variables are fixed. The trained model (“nn\_fit\_fol100” in this case, as we are
            predicting FOL links) is used to score the unclassified dataset. This scored model is stored in “nn\_predicted\_new”. Here               the scoring is done using the model trained on the complete dataset(62k SKUs)

        7.  The result (predicted FOL Link on unclassified data) is stored in “nn\_prob\_values”.

    2.  **Prediction for GPH** : Same steps as FOL link prediction

Result example,

| Prod_ID      | fol_predicted             | FOL_Ratio   | FOL_Prediction_confidence |
|--------------|---------------------------|-------------|---------------------------|
| Prod-1026102 | Kitchen Sinks             | 3.68E+22    | High Confidence           |
| Prod-1055452 | Iron Single Strap Saddles | 145247222.8 | Low Confidence            |
| Prod-1056364 | Kitchen Sinks             | 935857.2256 | Low Confidence            |

| Prod_ID      | gph_predicted                                                                                              | GPH_Ratio   | GPH_Prediction_confidence |
|--------------|------------------------------------------------------------------------------------------------------------|-------------|---------------------------|
| Prod-1026102 | 16 PLUMBING - FINISHED>>04 PLUMBING FIXTURES, PARTS AND ACCESSORIES>>19 COMMERCIAL SINKS>>08 KITCHEN SINKS | 7.84E+32    | Low Confidence            |
| Prod-1055452 | 15 PIPE, VALVES AND FITTINGS>>06 PIPE AND TUBING FITTINGS>>15 PVC PIPE FITTINGS>>18 PIPE WYES              | 1.03E+20    | Low Confidence            |
| Prod-1056364 | 16 PLUMBING - FINISHED>>04 PLUMBING FIXTURES, PARTS AND ACCESSORIES>>19 COMMERCIAL SINKS>>08 KITCHEN SINKS | 1.076828898 | Low Confidence            |

Please note: Differences in accuracy and high “Low Confidence” tagging in the result is due to the small sample data (200 rows) used.
