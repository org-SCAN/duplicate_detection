# Duplicate Detection

## Introduction

This project is a duplicate detection application designed to identify and analyze potential duplicates within datasets. It provides a variety of methods to compare records, handle different data types, and support customization through weighting and parameter selection. The application is built using Python and leverages libraries such as Streamlit for the user interface and various machine learning and data processing libraries for the comparison methods.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Adding a New Comparison Method](#adding-a-new-comparison-method)
  - [Method Structure](#method-structure)
  - [Example](#example)
- [Methods](#methods)
  - [Levenshtein Distance Method](#levenshtein-distance-method)
  - [Weighted Levenshtein Distance](#weighted-levenshtein-distance)
  - [Machine Learning Duplicate Detection](#machine-learning-duplicate-detection)
  - [Metaphone (Smooth Similarity)](#metaphone-smooth-similarity)
  - [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
  - [Fuzzy Matching](#fuzzy-matching)
  - [Embedding Similarity](#embedding-similarity)
  - [Composite Similarity](#composite-similarity)

---

## Getting Started

### Prerequisites

- **Python 3.7** or higher
- **pip** package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/duplicate-detection.git
   cd duplicate-detection
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit application by running:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

---

## Adding a New Comparison Method

You can extend the application by adding new comparison methods. Each method should follow the base structure and be placed in the `comparison_method` directory.

### Method Structure

Each comparison method is a class that inherits from `ComparisonMethodBase`. The class should implement the following methods:

- **`get_name()`**: A static method returning the name of the method as a string.
- **`get_parameter_info(data)`**: A static method returning a list of parameter definitions based on the provided data.
- **`__init__(**params)`**: The initializer that accepts parameters defined in `get_parameter_info`.
- **`compare(data, bdv_column)`**: The main method that performs the comparison and returns a pandas DataFrame with the results.

### Example

Here is a template for creating a new comparison method:

```python
from comparison_method.base import ComparisonMethodBase
import pandas as pd

class NewComparisonMethod(ComparisonMethodBase):
    @staticmethod
    def get_name():
        return "New Comparison Method"

    @staticmethod
    def get_parameter_info(data):
        # Define parameters based on the data
        parameter_info = [
            {
                'name': 'parameter_name',
                'type': 'parameter_type',  # e.g., 'select', 'multi_select', 'float', 'int'
                'options': ['option1', 'option2'],  # Only for 'select' or 'multi_select'
                'default': 'default_value',
                'description': 'Parameter description'
            },
            # Add more parameters as needed
        ]
        return parameter_info

    def __init__(self, **params):
        # Initialize parameters
        self.parameter_name = params.get('parameter_name', 'default_value')
        self.params = params  # Store all parameters if needed

    def compare(self, data, bdv_column):
        # Implement the comparison logic
        # For example, compare records and compute similarity scores

        # Prepare results
        result = pd.DataFrame({
            'Record 1': [],          # List of first records in pairs
            'Record 2': [],          # List of second records in pairs
            'Similarity (%)': []     # List of similarity scores
        })

        # Return the result DataFrame
        return result
```

**Steps to Add a New Method:**

1. **Create a New File:**

   - Save your new method class in a new Python file within the `comparison_method` directory.
   - The file name should be descriptive of the method (e.g., `new_comparison_method.py`).

2. **Implement the Required Methods:**

   - **`get_name()`**: Return the name that will appear in the application's method selection dropdown.
   - **`get_parameter_info(data)`**: Define the parameters your method requires, using the data to customize options if necessary.
   - **`__init__(**params)`**: Initialize your method with the parameters provided by the user.
   - **`compare(data, bdv_column)`**: Implement the logic to compare records and return the results.

3. **Update `__init__.py` (if necessary):**

   - Ensure that your new method is discoverable by the application. If using dynamic imports, this may not be necessary.

4. **Test Your Method:**

   - Run the application and verify that your new method appears in the method selection dropdown.
   - Test the method to ensure it works correctly with different datasets and parameter settings.

---

## Methods

Below are the detailed descriptions of each comparison method available in the application:

### [Levenshtein Distance Method](#levenshtein-distance-method)

#### **Description**

The **Levenshtein Distance Method** is a string similarity technique used to measure the difference between two sequences (typically strings). It calculates the minimum number of single-character edits required to change one word into the other. The allowed edits are:

- **Insertion**: Adding a character.
- **Deletion**: Removing a character.
- **Substitution**: Replacing one character with another.

In the context of duplicate detection, this method compares textual fields between records to identify potential duplicates based on how similar their text strings are. It is particularly effective for detecting typos, misspellings, and minor differences in text data.

#### **How It Works**

1. **Preprocessing**:
   - Convert text to lowercase to ensure case-insensitive comparison.
   - Remove or normalize special characters and whitespace if necessary.

2. **Calculating Levenshtein Distance**:
   - For each pair of records, compute the Levenshtein distance between the selected text fields.
   - The distance represents the number of edits needed to transform one string into the other.

3. **Converting Distance to Similarity Score**:
   - Convert the distance into a similarity percentage:
     \[
     \text{Similarity (\%)} = \left(1 - \frac{\text{Levenshtein Distance}}{\text{Maximum Length of the Two Strings}}\right) \times 100
     \]
   - A higher percentage indicates greater similarity.

4. **Thresholding**:
   - Define a similarity threshold (e.g., 80%).
   - Pairs with a similarity score above the threshold are considered potential duplicates.

#### **Advantages and Disadvantages**

| **Advantages**                                                                 | **Disadvantages**                                                        |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Simple and easy to implement and understand.                                   | Computationally intensive for large datasets (O(n²) comparisons).        |
| Effective at detecting typos and minor spelling errors.                        | Not suitable for non-textual or highly structured data.                  |
| Can handle strings of different lengths.                                       | Sensitive to character transpositions (e.g., "adn" vs. "and").           |
| No need for training data or complex parameter tuning.                         | Does not account for phonetic similarities or variations in word order.  |
| Useful for multilingual data as it operates at the character level.            | May not perform well with long strings due to cumulative differences.    |

#### **Suitable Data Types**

- **Textual Fields**:
  - Names (first names, last names).
  - Addresses.
  - Product names or descriptions.
  - Any other fields where minor typos or spelling variations are common.

### [Weighted Levenshtein Distance](#weighted-levenshtein-distance)

#### **Description**

The **Weighted Levenshtein Distance** method is an extension of the standard Levenshtein Distance algorithm, introducing weights to prioritize certain fields over others when comparing records. In duplicate detection, this method calculates a weighted average of the similarity scores between selected textual fields, allowing you to assign more importance to specific columns based on their relevance.

**How It Works:**

1. **Selection of Fields and Weights:**
   - Choose the text columns to compare and assign a weight to each one (ranging from 0 to 1).
   - Higher weights indicate greater importance in the overall similarity calculation.

2. **Preprocessing:**
   - Convert text to lowercase to ensure case-insensitive comparison.
   - Optionally, normalize text by removing special characters or whitespace.

3. **Calculating Levenshtein Distance for Each Field:**
   - For each pair of records, compute the Levenshtein distance between the corresponding fields.
   - Convert the distance into a similarity score for each field:
     \[
     \text{Field Similarity} = 1 - \frac{\text{Levenshtein Distance}}{\text{Maximum Length of the Two Strings}}
     \]
     - This yields a value between 0 and 1, where 1 indicates identical strings.

4. **Computing Weighted Similarity:**
   - Multiply each field's similarity score by its assigned weight.
   - Sum the weighted similarities and divide by the total weight to get the overall similarity score:
     \[
     \text{Overall Similarity} = \frac{\sum (\text{Field Similarity} \times \text{Field Weight})}{\sum \text{Field Weights}}
     \]
   - Convert the overall similarity score into a percentage by multiplying by 100.

5. **Thresholding and Results:**
   - Define a similarity threshold (e.g., 80%).
   - Pairs with an overall similarity score above the threshold are considered potential duplicates.
   - Results are sorted in descending order of similarity for review.

#### **Advantages and Disadvantages**

| **Advantages**                                                            | **Disadvantages**                                                        |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Allows customization by weighting fields based on their importance.       | Requires careful selection and tuning of weights for optimal results.    |
| More accurate than standard Levenshtein Distance when combining fields.   | Increased computational complexity due to multiple field comparisons.    |
| Can handle multiple text fields simultaneously.                           | Not suitable for non-textual or highly structured data without adaptation. |
| Effective at detecting duplicates when some fields are more critical.     | Still sensitive to character transpositions and does not account for phonetics. |
| No need for training data; straightforward to implement and understand.   | May not scale well with very large datasets without optimization.        |

#### **Suitable Data Types**

- **Textual Fields with Varying Importance:**
  - Names (assigning higher weight to last names than middle names).
  - Addresses (prioritizing street names over apartment numbers).
  - Product details where certain attributes are more significant.

#### **Use Cases**

- **Data Deduplication in Customer Databases:**
  - Emphasize fields like email addresses or phone numbers over less reliable fields.
  
- **Merging Product Catalogs:**
  - Assign higher weights to product IDs or SKUs compared to descriptions.

- **Record Linkage Across Data Sources:**
  - Adjust weights to reflect the reliability of fields from different sources.

By incorporating the Weighted Levenshtein Distance method, you gain the flexibility to fine-tune the duplicate detection process according to the specific importance of each field in your dataset. This method enhances accuracy by focusing on the most critical attributes, making it a valuable tool in scenarios where some data fields are more indicative of duplicates than others.

### [Machine Learning Duplicate Detection](#machine-learning-duplicate-detection)

#### Description

The **Machine Learning Duplicate Detection** method leverages machine learning algorithms to predict whether pairs of records are duplicates based on similarities and differences in their attributes. This approach constructs feature vectors for record pairs using selected text and numeric columns and trains a classifier to distinguish between duplicate and non-duplicate pairs.

**How It Works:**

1. **Selection of Columns:**

   - **Text Columns:** Choose textual fields to compute similarity scores (e.g., names, addresses).
   - **Numeric Columns:** Select numerical fields to compute differences (e.g., ages, prices).

2. **Feature Extraction:**

   - **Text Similarities:**
     - For each text column, compute TF-IDF vector representations of the texts.
     - Calculate the cosine similarity between the TF-IDF vectors of the two records.
   - **Numeric Differences:**
     - For each numeric column, compute the absolute difference between the values of the two records.

3. **Candidate Pair Generation:**

   - Generate all possible pairs of records to compare.

4. **Model Training:**

   - **Labeling:**
     - In practice, actual labels (duplicate or not) are required to train the model effectively.
     - In this implementation, synthetic labels are generated randomly for demonstration purposes.
   - **Training the Classifier:**
     - Split the data into training and testing sets.
     - Train a machine learning model (e.g., Logistic Regression) using the feature vectors and labels.
     - The model learns patterns that distinguish duplicates from non-duplicates.

5. **Prediction and Thresholding:**

   - Apply the trained model to the test set to obtain predicted probabilities for each pair.
   - Define a similarity threshold (e.g., 0.5).
   - Pairs with predicted probabilities above the threshold are considered potential duplicates.

6. **Result Compilation:**

   - Compile a results table listing the record pairs identified as potential duplicates, along with their similarity percentages.

#### Advantages and Disadvantages

| **Advantages**                                                            | **Disadvantages**                                                                    |
|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Captures complex patterns and relationships between features.             | Requires labeled data for meaningful training; synthetic labels are not reliable.    |
| Integrates multiple data types (text and numeric) seamlessly.             | Computationally intensive for large datasets due to pairwise comparisons.            |
| Model performance improves with more data and proper tuning.              | Risk of overfitting without proper validation and regularization.                    |
| Provides probabilistic outputs, allowing for adjustable thresholds.       | Generating all possible pairs is not scalable for very large datasets.               |
| Adaptable to different types of datasets and can evolve over time.        | Requires expertise in machine learning for effective implementation and tuning.      |

#### Suitable Data Types

- **Textual Fields:**

  - Names, addresses, product descriptions, comments.

- **Numeric Fields:**

  - Ages, prices, quantities, dates, or any numerical measurements where differences are meaningful.

### [Metaphone (Smooth Similarity)](#metaphone-smooth-similarity)

#### **Description**

The **Metaphone (Smooth Similarity)** method is a phonetic algorithm-based technique used for duplicate detection, especially effective for identifying words that sound similar but may be spelled differently. This method encodes words into phonetic representations (Metaphone codes) and compares them to detect similarities. It's particularly useful for handling variations in spelling, typos, and differences in pronunciation, making it ideal for names, addresses, and other textual data where phonetic similarity is important.

**How It Works:**

1. **Selection of Columns:**

   - Choose one or more text columns to compare using the Metaphone algorithm.
   - Assign weights to each selected column to indicate their importance in the overall similarity calculation.

2. **Preprocessing:**

   - Convert text to lowercase to ensure case-insensitive processing.
   - Remove or normalize special characters if necessary.

3. **Generating Metaphone Codes:**

   - For each textual value in the selected columns, compute its Metaphone code using the `phonetics` library.
   - The Metaphone algorithm transforms words into phonetic representations based on their pronunciation in English.

4. **Calculating Similarity:**

   - For each pair of records, compute the Levenshtein distance between their Metaphone codes for each selected column.
   - Convert the distance into a similarity score for each field:
     \[
     \text{Field Similarity (\%)} = \left(1 - \frac{\text{Levenshtein Distance}}{\text{Maximum Length of the Two Codes}}\right) \times 100
     \]
     - This yields a percentage between 0% and 100%, where 100% indicates identical phonetic codes.

5. **Computing Weighted Similarity:**

   - Multiply each field's similarity score by its assigned weight.
   - Sum the weighted similarities and divide by the total weight to obtain the overall similarity score:
     \[
     \text{Overall Similarity (\%)} = \frac{\sum (\text{Field Similarity} \times \text{Field Weight})}{\sum \text{Field Weights}}
     \]
     - This overall similarity score represents the phonetic similarity between two records.

6. **Thresholding and Results:**

   - Define a similarity threshold (e.g., 80%).
   - Pairs with an overall similarity score above the threshold are considered potential duplicates.
   - Results are sorted in descending order of similarity for review.

#### **Advantages and Disadvantages**

| **Advantages**                                                                 | **Disadvantages**                                                        |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Effective at identifying phonetic similarities and spelling variations.        | Designed primarily for English; less effective for other languages.      |
| Useful for detecting duplicates in names, addresses, and other text fields.    | May not handle homophones well (words that sound the same but are spelled differently). |
| Allows weighting of different fields based on importance.                      | Relies on the quality of the phonetic encoding; inaccuracies can affect results. |
| Enhances traditional string comparison methods by focusing on pronunciation.   | Computationally intensive for large datasets due to pairwise comparisons. |
| No need for training data; straightforward implementation.                     | Not suitable for numeric data or non-phonetic text.                      |

#### **Suitable Data Types**

- **Textual Fields Where Phonetics Matter:**

  - **Names:** First names, last names, middle names (e.g., "Steven" vs. "Stephen").
  - **Addresses:** Street names that may have spelling variations.
  - **Product Names:** Items where brand or product names might be misspelled.
  - **Other Text Data:** Any field where words may sound similar despite spelling differences.

By integrating the Metaphone (Smooth Similarity) method into your duplicate detection application, you enhance the ability to identify records that are phonetically similar. This method is particularly valuable when dealing with data entry errors, misspellings, or variations in spelling that standard string comparison methods might miss. It provides a more nuanced approach to textual similarity by considering how words sound, not just how they are spelled.

### [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)

#### **Description**

The **Locality-Sensitive Hashing (LSH)** method is an efficient algorithm for approximate nearest neighbor search in high-dimensional spaces. In the context of duplicate detection, LSH is used to quickly identify similar records by hashing them into buckets such that similar items are more likely to be in the same bucket. This method significantly reduces the number of comparisons needed, making it suitable for large datasets.

**How It Works:**

1. **Selection of Text Column:**

   - Choose a text column from the dataset to be used for hashing (e.g., product descriptions, customer comments).

2. **Tokenization:**

   - For each record, tokenize the text by converting it to lowercase and splitting it into words.

3. **MinHash Signature Generation:**

   - Create a MinHash signature for each record using the tokens.
   - The MinHash algorithm provides a compact representation of the set of tokens that preserves the Jaccard similarity between sets.

4. **Building the LSH Index:**

   - Use the MinHash signatures to build an LSH index.
   - Records with similar MinHash signatures are hashed into the same buckets with high probability.

5. **Querying for Similar Items:**

   - For each record, query the LSH index to find other records in the same bucket.
   - This step efficiently identifies candidate pairs of potential duplicates.

6. **Calculating Similarities:**

   - Compute the Jaccard similarity between the MinHash signatures of the candidate pairs.
   - Convert the similarity scores into percentages for interpretation.

7. **Thresholding and Results:**

   - Define a similarity threshold (e.g., 0.8).
   - Pairs with similarity scores above the threshold are considered potential duplicates.
   - Compile the results into a DataFrame, listing the record pairs and their similarity percentages.

#### **Advantages and Disadvantages**

| **Advantages**                                                         | **Disadvantages**                                                       |
|------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Highly efficient for large datasets due to reduced number of comparisons. | Only works with set-like data; primarily suitable for text data that can be tokenized. |
| Capable of handling high-dimensional data effectively.                 | May miss some duplicates due to the probabilistic nature of hashing (false negatives). |
| Scales well with the number of records.                                | Requires tuning of parameters like threshold and number of permutations. |
| Suitable for approximate nearest neighbor search.                      | Not as precise as exact similarity measures; may produce false positives. |
| Reduces computational complexity from O(n²) to approximately O(n).     | Implementation complexity is higher compared to simple distance measures. |

#### **Suitable Data Types**

- **Textual Data that Can Be Tokenized:**

  - Product descriptions.
  - Customer reviews or comments.
  - Articles or documents.
  - Any text field where the content can be meaningfully split into tokens (words).

#### **Use Cases**

- **Large-Scale Duplicate Detection:**

  - Identifying duplicate or near-duplicate records in big datasets where traditional pairwise comparison is computationally infeasible.

- **Plagiarism Detection:**

  - Finding similar documents or code snippets in large repositories.

- **Recommendation Systems:**

  - Finding items similar to a user's preferences based on text descriptions.

- **Document Clustering:**

  - Grouping similar documents together for topic modeling or data organization.

### [Fuzzy Matching](#fuzzy-matching)

#### **Description**

The **Fuzzy Matching** method is a string comparison technique used to measure the similarity between textual data, even when there are slight differences due to misspellings, typos, or variations in wording. This method leverages fuzzy string matching algorithms to compare text fields between records and determine how closely they match. It's particularly effective for identifying duplicates in datasets where exact string matches are unlikely due to inconsistencies in data entry.

**How It Works:**

1. **Selection of Text Columns:**

   - Choose one or more text columns from your dataset for comparison.
   - Assign weights to each selected column to indicate their importance in the overall similarity calculation.

2. **Generating Candidate Pairs:**

   - Generate all possible pairs of records to compare using combinations.
   - This step involves creating pairs without duplication to ensure each unique pair is evaluated once.

3. **Computing Fuzzy Similarity:**

   - For each pair of records and for each selected column:
     - Extract the textual values from both records.
     - Compute the fuzzy similarity score using the `fuzz.token_set_ratio` function from the `fuzzywuzzy` library.
       - This function calculates a similarity score between 0 and 100 based on the token set comparison of the two strings.
   - Multiply each similarity score by the assigned weight for that column.
   - Sum the weighted similarities and keep track of the total weight.

4. **Calculating Overall Similarity:**

   - For each pair, calculate the overall similarity score:
     \[
     \text{Overall Similarity (\%)} = \frac{\sum (\text{Field Similarity} \times \text{Field Weight})}{\sum \text{Field Weights}}
     \]
     - This results in a percentage between 0% and 100%, where higher percentages indicate greater similarity.

5. **Thresholding and Results:**

   - Define a similarity threshold (e.g., 80%).
   - Pairs with an overall similarity score equal to or above the threshold are considered potential duplicates.
   - Compile the results into a DataFrame, listing the record pairs and their similarity percentages.
   - Sort the results in descending order of similarity for easy review.

#### **Advantages and Disadvantages**

| **Advantages**                                                                 | **Disadvantages**                                                        |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Effective at identifying similar text despite typos, misspellings, or variations in wording. | Computationally intensive for large datasets due to pairwise comparisons. |
| Allows weighting of different fields based on their importance.                | May produce false positives if the similarity threshold is not set appropriately. |
| Utilizes advanced string matching algorithms for better accuracy.              | Sensitive to the choice of fuzzy matching function; different functions may yield different results. |
| No need for training data; straightforward to implement and interpret.         | Not suitable for numeric data or fields where token order is significant. |
| Can handle multi-word strings and partial matches effectively.                 | Scaling issues with large datasets; may require optimization techniques like blocking or indexing. |

#### **Suitable Data Types**

- **Textual Fields with Potential Variations:**

  - **Names:** First names, last names, full names where spelling variations may occur.
  - **Addresses:** Street addresses, city names, or postal information that may have inconsistencies.
  - **Product Names or Descriptions:** Items where descriptions may vary slightly.
  - **Comments or Notes:** Unstructured text data containing valuable information.

#### **Use Cases**

- **Customer Data Deduplication:**

  - Identifying duplicate customer records where names or addresses have slight differences.

- **Product Catalog Cleaning:**

  - Matching products with similar names or descriptions that are not exact matches.

- **Data Integration:**

  - Merging datasets from different sources where field values may not align perfectly.

- **Error Correction:**

  - Detecting and correcting typos or misspellings in textual data.


By incorporating the Fuzzy Matching method into your duplicate detection application, you enhance the ability to identify records that are similar but not identical, due to variations in text data. This method provides a flexible and effective approach to handling inconsistencies commonly found in real-world datasets.

### [Embedding Similarity](#embedding-similarity)

#### **Description**

The **Embedding Similarity** method leverages advanced language models to convert textual data into numerical vector representations known as embeddings. This approach captures the semantic meaning of text, allowing for more nuanced comparisons between records beyond simple lexical matching. By transforming both text and numeric data into a unified embedding space, this method facilitates the identification of duplicates based on the overall similarity of records.

**How It Works:**

1. **Selection of Columns:**

   - **Text Columns:** Choose textual fields to be embedded using a pre-trained language model.
   - **Numeric Columns:** Select numerical fields to include in the similarity calculation.

2. **Embedding Text Data:**

   - Use a pre-trained Sentence Transformer model (e.g., `'all-MiniLM-L6-v2'`) to generate embeddings for each selected text column.
   - Multiply each text column's embeddings by a user-defined weight to adjust its importance in the overall similarity.

3. **Processing Numeric Data:**

   - Standardize numeric columns using `StandardScaler`.
   - Apply user-defined weights to each numeric column to reflect its significance.

4. **Combining Embeddings and Numeric Data:**

   - Concatenate the weighted text embeddings and numeric data to form a combined feature vector for each record.

5. **Computing Distance Matrix:**

   - Calculate pairwise distances between records using a specified distance metric (e.g., cosine, Euclidean, Manhattan, Chebyshev).
   - For cosine distance, normalize embeddings to ensure accurate computation.

6. **Similarity Calculation:**

   - Convert distances into similarity scores, normalized between 0% and 100%.
   - A higher similarity score indicates greater similarity between records.

7. **Clustering (Optional):**

   - Apply a clustering algorithm (e.g., KMeans, Agglomerative Clustering, DBSCAN, OPTICS) to group similar records.
   - Clustering helps identify groups of potential duplicates.

8. **Thresholding and Results:**

   - Compile the results into a DataFrame, listing record pairs and their similarity percentages.
   - Sort the results in descending order of similarity for easy review.

#### **Advantages and Disadvantages**

| **Advantages**                                                              | **Disadvantages**                                                        |
|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Captures semantic similarity between texts, not just surface-level matches. | Computationally intensive, especially for large datasets.                |
| Integrates both text and numeric data into a unified similarity measure.    | Requires a pre-trained language model, which may not capture all nuances. |
| Effective at detecting duplicates with different wording but similar meaning. | Embedding generation can be time-consuming.                              |
| Flexible with customizable weights and choice of distance metrics.          | High-dimensional embeddings may be difficult to interpret directly.       |
| Supports various clustering algorithms for grouping similar records.        | Performance depends on the quality of the embeddings and model used.     |

#### **Suitable Data Types**

- **Textual Fields:**

  - Product descriptions.
  - Customer reviews or comments.
  - Any text where semantic meaning is important.

- **Numeric Fields:**

  - Prices, quantities, or measurements.
  - Numerical attributes that contribute to record similarity.

#### **Use Cases**

- **Product Catalog Matching:**

  - Identifying duplicate or similar products across different catalogs based on descriptions and specifications.

- **Customer Data Deduplication:**

  - Merging customer records where names, addresses, or notes may vary in wording but refer to the same individual.

- **Content Recommendation:**

  - Grouping similar articles, documents, or media based on semantic content for recommendation systems.

- **Data Integration:**

  - Aligning records from disparate datasets where textual fields may not exactly match but are semantically equivalent.

### [Composite Similarity](#composite-similarity)

#### **Description**

The **Composite Similarity** method combines multiple similarity measures across different data types—strings, numbers, and dates—to compute an overall similarity score between records. This comprehensive approach allows for more accurate duplicate detection by considering various attributes and their respective data types within the records.

**How It Works:**

1. **Selection of Columns:**

   - **String Columns:** Choose textual fields to compare using the **Levenshtein distance**.
   - **Numeric Columns:** Select numerical fields to compare based on normalized differences.
   - **Date Columns:** Choose date fields to compare based on the difference in days.

2. **Weight Assignment:**

   - Assign weights to each selected column to indicate its importance in the overall similarity calculation.
   - Weights range from 0 to 1, with higher weights increasing a column's influence on the final score.

3. **Generating Candidate Pairs:**

   - Generate all unique pairs of records for comparison.
   - This ensures that each pair is evaluated once without redundancy.

4. **Computing Similarities:**

   - **String Columns:**
     - For each string column:
       - Extract the textual values from both records.
       - Compute the Levenshtein distance between the two strings.
       - Convert the distance into a similarity score:
         \[
         \text{Similarity} = 1 - \frac{\text{Levenshtein Distance}}{\text{Maximum Length of the Two Strings}}
         \]
       - If both strings are empty, set similarity to 1 (identical).

   - **Numeric Columns:**
     - For each numeric column:
       - Calculate the absolute difference between the two values.
       - Normalize the difference by the range of the column:
         \[
         \text{Normalized Difference} = \frac{|\text{Value}_1 - \text{Value}_2|}{\text{Max Value} - \text{Min Value} + \epsilon}
         \]
         - \(\epsilon\) is a small constant (e.g., \(1 \times 10^{-5}\)) to avoid division by zero.
       - Compute the similarity:
         \[
         \text{Similarity} = 1 - \text{Normalized Difference}
         \]
       - Values closer together have higher similarity.

   - **Date Columns:**
     - For each date column:
       - Parse the date strings into date objects.
       - Calculate the absolute difference in days between the two dates.
       - Normalize the difference by the total date range in the column:
         \[
         \text{Normalized Difference} = \frac{|\text{Date}_1 - \text{Date}_2|}{\text{Max Date} - \text{Min Date} + \epsilon}
         \]
       - Compute the similarity:
         \[
         \text{Similarity} = 1 - \text{Normalized Difference}
         \]
       - If date parsing fails, set similarity to 0.

5. **Calculating Weighted Overall Similarity:**

   - Multiply each field's similarity score by its assigned weight.
   - Sum the weighted similarities and divide by the total weight:
     \[
     \text{Overall Similarity (\%)} = \left( \frac{\sum (\text{Field Similarity} \times \text{Field Weight})}{\sum \text{Field Weights}} \right) \times 100
     \]
   - The result is a percentage between 0% and 100%, where higher percentages indicate greater overall similarity.

6. **Thresholding and Results:**

   - Compile the results into a DataFrame, listing record pairs and their similarity percentages.
   - Sort the results in descending order of similarity for easy review.
   - Use similarity scores to identify potential duplicates based on a chosen threshold.

#### **Advantages and Disadvantages**

| **Advantages**                                                                | **Disadvantages**                                                        |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Integrates multiple data types (strings, numbers, dates) into one similarity measure. | Computationally intensive for large datasets due to pairwise comparisons. |
| Allows weighting of fields based on their importance.                         | Requires careful tuning of weights and thresholds for optimal results.   |
| Provides a comprehensive assessment of record similarity.                     | Date parsing may fail if date formats are inconsistent or invalid.       |
| No need for training data; straightforward to implement and interpret.        | Sensitive to outliers in numeric data due to normalization by range.     |
| Can handle missing values by adjusting similarity calculations.               | Scaling issues with very large datasets; may need optimization techniques. |

#### **Suitable Data Types**

- **String Fields:**

  - Names, addresses, descriptions, or any textual data where spelling variations may occur.

- **Numeric Fields:**

  - Ages, prices, quantities, measurements, or any numerical attributes where value differences are meaningful.

- **Date Fields:**

  - Dates of birth, transaction dates, registration dates, or any fields containing date information.

#### **Use Cases**

- **Customer Data Deduplication:**

  - Identifying duplicate customer records by comparing names, dates of birth, contact information, and customer IDs.

- **Product Data Matching:**

  - Merging product records by comparing product names, prices, launch dates, and other attributes.

- **Data Integration:**

  - Combining datasets from different sources where fields may have varying data types and formats, requiring a holistic similarity measure.

- **Record Linkage Across Systems:**

  - Linking records from different databases or systems using a composite of textual, numerical, and date attributes to find matching entities.

By utilizing the Composite Similarity method, you can achieve a more nuanced and accurate duplicate detection process. This method is particularly beneficial when dealing with complex datasets that include a mix of text, numeric, and date fields, allowing you to tailor the importance of each field according to your specific needs.