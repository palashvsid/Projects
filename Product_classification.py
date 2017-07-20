# -----------------PRODUCT CATEGORIZATION CODE---------------------------------

#Importing the required packages, use pip/conda to install packages if not already installed
import re
import string
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Setting working directory
os.chdir('PLEASE SET WORKING DIRECTORY AND HIT RUN')

# Reading the enriched data#
data = pd.read_csv("Sample.csv",encoding="latin")

# Reading the mapping dictionary#
mapping_dict = pd.read_csv("abbrevation_mapping_dict.csv",encoding="latin")


# ---------------------------------USER DEFINED FUNCTIONS------------------------

# Function to remove the last two pipe entries from the GPH path#
def remove_last_two_pipe(text, separator, sep_to_remove):
    temp = text.split(separator)
    max_len = len(temp)

    if text.endswith(separator):
        to_remove = sep_to_remove + 1
    else:
        to_remove = sep_to_remove

    val_to_return = '|'. join(temp[0:(max_len - to_remove)])
    return val_to_return


# Function to clean the text--- Remove special characters #
def clean_text(text):
    text = text.lower()
    chars = re.escape(string.punctuation)
    text = re.sub(r'[' + chars + ']', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r' ', '_', text)

    return text

def replace_space(text):
    return text.replace(" ", "_")


# Function to look up the values from dictionary #
def map_abbreviation(text, dictionary):
    text = str(text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    split_text = text.split()

    base_words = pd.DataFrame(split_text)
    base_words.columns = ['ABBREVIATION']
    t = mapping_dict[mapping_dict.ABBREVIATION.isin(split_text)].reset_index()

    merge_table = base_words.merge(
        t[['ABBREVIATION', 'DESCRIPTION']], how="left")

    merge_table['cleaned'] = merge_table['DESCRIPTION']

    merge_table.loc[merge_table.cleaned.isnull(),
                    'cleaned'] = merge_table.loc[merge_table.cleaned.isnull(),
                                                 'ABBREVIATION']

    text_to_return = str(' '.join(merge_table['cleaned']).encode('utf-8'))

    return text_to_return


# Function to remove duplicates from the data
def remove_duplicate_data(data, list_of_idv, dv):
    remove_data = data.groupby(list_of_idv,
                               as_index=False).agg({dv: pd.Series.nunique})
    excluded_rows = pd.merge(data,
                             remove_data[remove_data[dv] > 1],
                             how='right',
                             left_on=list_of_idv,
                             right_on=list_of_idv)
    included_rows = pd.merge(data,
                             remove_data[remove_data[dv] == 1],
                             how='left',
                             left_on=list_of_idv,
                             right_on=list_of_idv)

    included_final = included_rows[included_rows[dv + "_y"] == 1]
    included_final = included_final.drop(dv + "_y", axis=1)

    included_final = included_final.rename(columns={dv + "_x": dv})

    return included_final, excluded_rows


# Function to create unigram-bigram matrix
def create_tdm_dataframe(data, column_name, num_words, suffix):

    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(ngram_range=(1, num_words))
    vect.fit(data[column_name])
    train_dtm = vect.transform(data[column_name])

    import pandas as pd
    dtm_data = pd.DataFrame(
        train_dtm.toarray(),
        columns=vect.get_feature_names())
    new_names = [(i, i + '_' + suffix) for i in dtm_data.columns]
    dtm_data.rename(columns=dict(new_names), inplace=True)

    return(dtm_data)


# Funtion to calculate threshold value
def ratio_threshold(probabilities, prediction, expected, model_classes):
    collated_df = pd.DataFrame(probabilities, columns=model_classes.classes_)
    predicted_classes = pd.DataFrame(prediction, columns=["predicted_class"])
    actual_classes = pd.DataFrame(expected.reset_index())
    actual_classes.columns.values[1] = "Actual_class"

    collated_df = pd.concat(
        [collated_df, predicted_classes, actual_classes["Actual_class"]], axis=1)

    # Deriving the max and 2nd max values along each row #
    collated_df1 = collated_df.drop(collated_df.columns[-2:], axis=1)
    m = pd.DataFrame(np.sort(collated_df1.values)[
                     :, -2:], columns=['2nd', '1st'])

    # Calculating the ratio of max/2nd max #
    m["Ratio"] = m.apply(lambda row: row['1st'] / row['2nd'], axis=1)
    collated_df = pd.concat([collated_df, m["Ratio"]], axis=1)

    # Assigning flag 1 or 0 based on the prediction
    collated_df["flag"] = collated_df.apply(lambda row: 1 if(
        row["predicted_class"] == row["Actual_class"]) else 0, axis=1)

    # Forming dataframe with only the required columns for threshold
    # calculation
    threshold_df = pd.concat(
        [collated_df["Ratio"], collated_df["flag"], collated_df["Ratio"]], axis=1)
    threshold_df.columns.values[2] = 'Ratio1'

    # Grouping by Ratio to get sum of right predictions and count of ratio
    threshold_df1 = threshold_df.groupby('Ratio').agg(
        {'Ratio1': 'count', 'flag': 'sum'}).reset_index().rename(columns={'Ratio1': 'Ratio Count'})

    # Sorting by Ratio in descending order
    threshold_df1 = threshold_df1.sort_values(['Ratio'], ascending=[False])

    # Deriving the cumulative true prediction and cumulative total values
    threshold_df1["total"] = np.cumsum(threshold_df1["Ratio Count"])
    threshold_df1["true"] = np.cumsum(threshold_df1["flag"])

    # Calculating the cumulative percentage
    threshold_df1["Cumulative_%"] = threshold_df1.apply(
        lambda row: row["true"] / row["total"], axis=1)
    # Filtering data for percentage >=0.99
    filtered_threshold = threshold_df1.loc[threshold_df1['Cumulative_%'] >= 0.99]

    # Deriving the threshold value
    thresholdValue = filtered_threshold["Ratio"].min()
    return(thresholdValue)


# --------------------------------DATA PREPARATION------------------------
# Clean the column names #
data.columns = map(clean_text, data.columns)
data.columns = map(replace_space, data.columns)

# Remove the last two pipes in the gph full path #
data['gph_cleaned'] = data['gph_full_path'].apply(
    lambda x: remove_last_two_pipe(x, "|", 2))


# Expand the prod description and prod long description based on the mapping #
data['prd_desc_expanded'] = data['productdesc'].apply(
    lambda x: map_abbreviation(x, mapping_dict))

# Now map the long description columns as well with the fol link #

# Create a new column for long desc and then map the abbrevations -- if
# long desc is null then  use prd desc #
data['prd_long_desc_new'] = data['prodlongdesc']
data.loc[data.prodlongdesc.isnull(),
         'prd_long_desc_new'] = data.loc[data.prodlongdesc.isnull(),
                                         'productdesc']


data['prd_long_desc_expanded'] = data['prd_long_desc_new'].apply(
    lambda x: map_abbreviation(x, mapping_dict))

# Removing SKU's with gph_full path as Onboarding #
final_data = data[~data['gph_full_path'].str.contains("Onboarding")]


# -------------------------------------Product Categorization for GPH-------------------------
# ---------------------------------------DOCUMENT TERM MATRIX------------------------------
# Deriving list of independent and dependent variables
list_of_idv = [
    "productdesc",
    "primary_vendor",
    "discgroupiddesc",
    "linebuyiddesc"]

dv = 'gph_cleaned'

# Code to remove misclassified entires from the data #
gph_final_data, gph_removed_data = remove_duplicate_data(
    final_data, list_of_idv, dv)

# Creating unigrams & bigrams matrix#
# Product long description tdm matrix #
dtm_data1 = create_tdm_dataframe(
    gph_final_data, 'prd_long_desc_expanded', 2, 'desc')

#disc group tdm matrix #
dtm_data2 = create_tdm_dataframe(gph_final_data, 'discgroupiddesc', 1, "disc")

# Line buy tdm matrix #
dtm_data4 = create_tdm_dataframe(gph_final_data, 'linebuyiddesc', 1, "linebuy")

# Creating vendor dummies matrix
vendor_dummies = pd.get_dummies(gph_final_data['primary_vendor'])
new_names = [(i, i + '_vendor') for i in vendor_dummies.columns]
vendor_dummies.rename(columns=dict(new_names), inplace=True)


# Compiling the unigrams and bigrams into a single dataset
dtm_data_gph = pd.concat([dtm_data1.reset_index(drop=True),
                          dtm_data2.reset_index(drop=True),
                          dtm_data4.reset_index(drop=True),
                          vendor_dummies.reset_index(drop=True)],
                         axis=1)


dtm_data_gph['gph_cleaned'] = gph_final_data['gph_cleaned'].reset_index()[
    'gph_cleaned']


# Renaming column named fit, to avoid compiler discrepency
dtm_data_gph = dtm_data_gph.rename(columns={'fit': 'fit_feature'})


# Remove single sku classes #
single_sku_classes = dtm_data_gph['gph_cleaned'].value_counts()[
    dtm_data_gph['gph_cleaned'].value_counts() == 1].index.values.tolist()
dtm_data_gph = dtm_data_gph[~dtm_data_gph['gph_cleaned'].isin(
    single_sku_classes)]

# document term matrix has been created #


# -----------------------------Neural Network Model------------------------

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(
        80,
        60),
    random_state=1234)

# Creating train and test split
from sklearn.model_selection import train_test_split
X_train_gph, X_test_gph, Y_train_gph, Y_test_gph = train_test_split(
    dtm_data_gph.iloc[:, dtm_data_gph.columns != 'gph_cleaned'], 
    dtm_data_gph['gph_cleaned'], test_size=0.20, random_state=1234)

# Training the model for GPH predictions using 80% dataset
nn_fitted_gph = model.fit(X_train_gph, Y_train_gph)


# Prediction and accuracy #
predicted_nn_gph = nn_fitted_gph.predict(X_test_gph)
prob_nn_gph = nn_fitted_gph.predict_proba(X_test_gph)

from sklearn import metrics
gph_accuracy = metrics.accuracy_score(Y_test_gph, predicted_nn_gph)
print(gph_accuracy)


# -------------------------------------THRESHOLD VALUE CALCUATION------------------------
# threshold is calculated based on the user defined function
gph_threshold = ratio_threshold(
    prob_nn_gph,
    predicted_nn_gph,
    Y_test_gph,
    nn_fitted_gph)


# -----------------------------PRODUCT CLASSIFICATION FOR FOL LINK-----------------------
# ---------------------------------DOCUMENT TERM MATRIX-----------------------------------:


list_of_idv = [
    "productdesc",
    "primary_vendor",
    "discgroupiddesc",
    "linebuyiddesc"]

dv = 'fol_link'

# Code to remove misclassified entires from the data
gph_final_data, gph_removed_data = remove_duplicate_data(
    final_data, list_of_idv, dv)

# Creating unigrams & bigrams matrix #
# Product long description tdm matrix #
dtm_data1 = create_tdm_dataframe(
    gph_final_data, 'prd_long_desc_expanded', 2, 'desc')

#disc group tdm matrix #
dtm_data2 = create_tdm_dataframe(gph_final_data, 'discgroupiddesc', 1, "disc")

# Line buy tdm matrix #
dtm_data4 = create_tdm_dataframe(gph_final_data, 'linebuyiddesc', 1, "linebuy")

# Creating vendor dummies matrix
vendor_dummies = pd.get_dummies(gph_final_data['primary_vendor'])
new_names = [(i, i + '_vendor') for i in vendor_dummies.columns]
vendor_dummies.rename(columns=dict(new_names), inplace=True)


# Compiling the unigrams and bigrams into a single dataset #
dtm_data_fol = pd.concat([dtm_data1.reset_index(drop=True),
                          dtm_data2.reset_index(drop=True),
                          dtm_data4.reset_index(drop=True),
                          vendor_dummies.reset_index(drop=True)],
                         axis=1)

# Assigning expected classifications to the dataset
dtm_data_fol['fol_link'] = gph_final_data['fol_link'].reset_index()[
    'fol_link']


# Renaming column named fit, to avoid compiler discrepency #
dtm_data_fol = dtm_data_fol.rename(columns={'fit': 'fit_feature'})


# Remove single sku classes #
single_sku_classes = dtm_data_fol['fol_link'].value_counts()[
    dtm_data_fol['fol_link'].value_counts() == 1].index.values.tolist()
dtm_data_fol = dtm_data_fol[~dtm_data_fol['fol_link'].isin(single_sku_classes)]


# ----------------------------------Neural Network Model -----------------------------#

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(
        80,
        60),
    random_state=1234)

from sklearn.model_selection import train_test_split
X_train_fol, X_test_fol, Y_train_fol, Y_test_fol = train_test_split(
    dtm_data_fol.iloc[:, dtm_data_fol.columns != 'fol_link'], 
    dtm_data_fol['fol_link'], 
    test_size=0.20, random_state=1234)

# Training the model for FOL predictions on 80% data
nn_fitted_fol = model.fit(X_train_fol, Y_train_fol)


# Prediction and accuracy #
predicted_nn_fol = nn_fitted_fol.predict(X_test_fol)
prob_nn_fol = nn_fitted_fol.predict_proba(X_test_fol)

from sklearn import metrics
fol_accuracy = metrics.accuracy_score(Y_test_fol, predicted_nn_fol)
print(metrics.accuracy_score(Y_test_fol, predicted_nn_fol))


# ----------------------------THRESHOLD VALUE CALCUATION-------------------------------

# Threshold calculated using the defined function
fol_threshold = ratio_threshold(
    prob_nn_fol,
    predicted_nn_fol,
    Y_test_fol,
    nn_fitted_fol)


# ----------------------------- PRODUCT CLASSIFICATION ON UNCLASSIFIED SKU's---------------------
# -----------------------------------------------------------Prediction for FOL
# Selecting the unclassified SKU's
final_data1 = data[data['gph_full_path'].str.contains("Onboarding")]


# Setting the independent and dependent variables
list_of_idv = [
    "productdesc",
    "primary_vendor",
    "discgroupiddesc",
    "linebuyiddesc"]

dv = 'fol_link'

# Remove duplicates
gph_final_data, gph_removed_data = remove_duplicate_data(
    final_data1, list_of_idv, dv)


# Product long description tdm matrix #
dtm_data1 = create_tdm_dataframe(
    gph_final_data, 'prd_long_desc_expanded', 2, 'desc')

#disc group tdm matrix #
dtm_data2 = create_tdm_dataframe(gph_final_data, 'discgroupiddesc', 1, "disc")

# Line buy tdm matrix #
dtm_data4 = create_tdm_dataframe(gph_final_data, 'linebuyiddesc', 1, "linebuy")

# Creating venfor dummies matrix
vendor_dummies = pd.get_dummies(gph_final_data['primary_vendor'])
new_names = [(i, i + '_vendor') for i in vendor_dummies.columns]
vendor_dummies.rename(columns=dict(new_names), inplace=True)

# Compiling the document term matrix
dtm_data = pd.concat([dtm_data1.reset_index(drop=True),
                      dtm_data2.reset_index(drop=True),
                      dtm_data4.reset_index(drop=True),
                      vendor_dummies.reset_index(drop=True)],
                     axis=1)

# Creating list of columns from the model
col_list_model = pd.DataFrame(X_train_fol.columns.values)

# Renaming the column
col_list_model = col_list_model.rename(columns={0: 'col_names'})

# Deriving the set of availabe and required columns in the document term matrix
# So that the trained model has the same columns as document term matrix
present_col = set(dtm_data.columns)
required_col = set(col_list_model['col_names'])

# create a new data frame to test
test_dtm = pd.DataFrame()

common_cols = present_col.intersection(required_col)
differ_cols = required_col.difference(present_col)
len(common_cols)

# add the common columns to the test data frame #
for cols in common_cols:
    test_dtm[cols] = dtm_data[cols]


# add the differ columns as 0 in the test data frame #
for cols in differ_cols:
    test_dtm[cols] = 0


# Adding FOL link, the expected class to the matrix
test_dtm['fol_link'] = gph_final_data['fol_link'].reset_index()['fol_link']

# Deriving independent and dependent variables for scoring
X_test_fol1 = test_dtm[col_list_model['col_names']]
Y_test_fol1 = test_dtm['fol_link']


col_list_model.loc[col_list_model.index.max() + 1] = ['fol_link']

# Scoring using the trained model,
# Here the model is trained on the entire dataset

# Deleting and redefining the model to avoid discrepency in scoring
del model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(
        80,
        60),
    random_state=1234)


# Training the model on entire dataset for fol
nn_fit_fol100 = model.fit(dtm_data_fol.drop(
    dtm_data_fol.columns[-1:], axis=1), dtm_data_fol['fol_link'])

# Training model on entire dataset
# Use the following code model for GPH scoring,

# nn_fit_gph100 = model.fit(dtm_data_gph.drop(dtm_data_gph.columns[-1:], axis=1), 
# dtm_data_gph['gph_cleaned'])

# Deriving predictions based on the learnt model
nn_predicted_new = nn_fit_fol100.predict(X_test_fol1)
from sklearn import metrics


# Deriving the accuracy
print(metrics.accuracy_score(Y_test_fol1, nn_predicted_new))

# Compiling the unclassified predictions
nn_predicted_proba = nn_fit_fol100.predict_proba(X_test_fol1)
nn_prob_values = pd.DataFrame(nn_predicted_proba)
nn_prob_values['fol_predicted'] = nn_predicted_new
nn_prob_values['expected'] = Y_test_fol1
nn_prob_values['Prod_ID'] = gph_final_data['id']

# Deriving the 1st and 2nd max ratio for each SKU

# Deriving the 1st and 2nd max through sorting each row
nn_max = nn_prob_values.drop(nn_prob_values.columns[-3:], axis=1)
temp_nn = pd.DataFrame(np.sort(nn_max.values)[:, -2:], columns=['2nd', '1st'])

# Calculating the ratio of max/2nd max #
temp_nn["FOL_Ratio"] = temp_nn.apply(lambda row: row['1st'] / row['2nd'], axis=1)
nn_prob_values = pd.concat([nn_prob_values, temp_nn["FOL_Ratio"]], axis=1)

# Assigning Prediction confidence
nn_prob_values["FOL_Prediction_confidence"] = nn_prob_values.apply(
    lambda row: "High Confidence" if(
        row["FOL_Ratio"] >= fol_threshold) else "Low Confidence", axis=1)


# ---------------------------------------------------Prediction for GPH

# Setting the independent and dependent variables
list_of_idv = [
    "productdesc",
    "primary_vendor",
    "discgroupiddesc",
    "linebuyiddesc"]

dv = 'gph_cleaned'

# Remove duplicates
gph_final_data, gph_removed_data = remove_duplicate_data(
    final_data1, list_of_idv, dv)


# Product long description tdm matrix #
dtm_data1 = create_tdm_dataframe(
    gph_final_data, 'prd_long_desc_expanded', 2, 'desc')

#disc group tdm matrix #
dtm_data2 = create_tdm_dataframe(gph_final_data, 'discgroupiddesc', 1, "disc")

# Line buy tdm matrix #
dtm_data4 = create_tdm_dataframe(gph_final_data, 'linebuyiddesc', 1, "linebuy")

# Creating venfor dummies matrix
vendor_dummies = pd.get_dummies(gph_final_data['primary_vendor'])
new_names = [(i, i + '_vendor') for i in vendor_dummies.columns]
vendor_dummies.rename(columns=dict(new_names), inplace=True)

# Compiling the document term matrix
dtm_dataG = pd.concat([dtm_data1.reset_index(drop=True),
                      dtm_data2.reset_index(drop=True),
                      dtm_data4.reset_index(drop=True),
                      vendor_dummies.reset_index(drop=True)],
                     axis=1)

# Creating list of columns from the model
col_list_model = pd.DataFrame(X_train_gph.columns.values)

# Renaming the column
col_list_model = col_list_model.rename(columns={0: 'col_names'})

# Deriving the set of availabe and required columns in the document term matrix
# So that the trained model has the same columns as document term matrix
present_col = set(dtm_dataG.columns)
required_col = set(col_list_model['col_names'])

# create a new data frame to test
test_dtm1 = pd.DataFrame()

common_cols = present_col.intersection(required_col)
differ_cols = required_col.difference(present_col)
len(common_cols)

# add the common columns to the test data frame #
for cols in common_cols:
    test_dtm1[cols] = dtm_dataG[cols]


# add the differ columns as 0 in the test data frame #
for cols in differ_cols:
    test_dtm1[cols] = 0


# Adding FOL link, the expected class to the matrix
test_dtm1['gph_cleaned'] = gph_final_data['gph_cleaned'].reset_index()['gph_cleaned']

# Deriving independent and dependent variables for scoring
X_test_gph1 = test_dtm1[col_list_model['col_names']]
Y_test_gph1 = test_dtm1['gph_cleaned']


col_list_model.loc[col_list_model.index.max() + 1] = ['gph_cleaned']

# Scoring using the trained model,
# Here the model is trained on the entire dataset
# Training the model on entire dataset for fol
nn_fit_gph100 = model.fit(dtm_data_gph.drop(dtm_data_gph.columns[-1:], axis=1), 
                          dtm_data_gph['gph_cleaned'])

# Deriving predictions based on the learnt model
gph_predicted_new = nn_fit_gph100.predict(X_test_gph1)
from sklearn import metrics


# Deriving the accuracy
#print(metrics.accuracy_score(Y_test_fol1, nn_predicted_new))

# Compiling the unclassified predictions
gph_predicted_proba = nn_fit_gph100.predict_proba(X_test_gph1)
gph_prob_values = pd.DataFrame(gph_predicted_proba)
gph_prob_values['gph_predicted'] = gph_predicted_new
gph_prob_values['Prod_ID'] = gph_final_data['id']

# Deriving the 1st and 2nd max ratio for each SKU
# Deriving the 1st and 2nd max through sorting each row
gph_max = gph_prob_values.drop(gph_prob_values.columns[-3:], axis=1)
temp_gph = pd.DataFrame(np.sort(gph_max.values)[:, -2:], columns=['2nd', '1st'])

# Calculating the ratio of max/2nd max #
temp_gph["GPH_Ratio"] = temp_gph.apply(lambda row: row['1st'] / row['2nd'], axis=1)
gph_prob_values = pd.concat([gph_prob_values, temp_gph["GPH_Ratio"]], axis=1)

# Assigning Prediction confidence
gph_prob_values["GPH_Prediction_confidence"] = gph_prob_values.apply(
    lambda row: "High Confidence" if(
        row["GPH_Ratio"] >= gph_threshold) else "Low Confidence", axis=1)

# Final Output

output = pd.concat([nn_prob_values["Prod_ID"],
                     nn_prob_values["fol_predicted"],
                     gph_prob_values["gph_predicted"],
                     nn_prob_values["FOL_Ratio"],
                     gph_prob_values["GPH_Ratio"],
                     nn_prob_values["FOL_Prediction_confidence"],
                     gph_prob_values["GPH_Prediction_confidence"]],
                    axis=1)

# Writing the output files into the set directory

output.to_csv('output.csv')
