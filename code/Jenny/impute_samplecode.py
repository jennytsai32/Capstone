def impute_numeric(df, strategy='mean'):
    """
    Impute missing numeric values in df based on strategy ('mean', 'median', 'mode', 'max')
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df[col].fillna((df[col].mode()), inplace=True)
        elif strategy == 'max':
            df[col].fillna(df[col].max(), inplace=True)
        else:
            print('Invalid imputation strategy. Using mean instead.')
            df[col].fillna(df[col].mean(), inplace=True)

    return df
# ----------------------------------------------------------------------------------------------------------------------

def impute_boolean(df, strategy='most_frequent'):
    """
    Impute missing values in boolean columns
    """

    bool_cols = []

    for col in df.columns:
        uniques = df[col].unique()
        has_nan = pd.isna(uniques).any()

        if has_nan:
            # Make sure NaN itself is not considered unique
            uniques = uniques[~pd.isna(uniques)]

        # Check 0s and 1s or Trues and Falses
        if set(uniques) <= {0, 1} or set(uniques) <= {True, False}:
            bool_cols.append(col)

    for col in bool_cols:

        if strategy == 'most_frequent':
            most_freq = df[col].mode()[0]
            df[col].fillna(most_freq, inplace=True)

        elif strategy == 'all_true':
            df[col] = df[col].astype(bool)
            df[col].fillna(True, inplace=True)

        elif strategy == 'all_false':
            df[col] = df[col].astype(bool)
            df[col].fillna(False, inplace=True)

        else:
            print('Invalid strategy. Using most frequent.')
            most_freq = df[col].mode()[0]
            df[col].fillna(most_freq, inplace=True)

    return df
# ----------------------------------------------------------------------------------------------------------------------


def remove_all_nan_columns(df):
    """
    Remove columns with all NaN values from a DataFrame
    """
    # Get columns with all NaNs
    all_nan_cols = [col for col in df.columns if df[col].isnull().all()]

    # Drop columns with all NaNs
    df.drop(all_nan_cols, axis=1, inplace=True)

    return df

# ----------------------------------------------------------------------------------------------------------------------


def impute_all_categorical(df):
    """
    Impute missing values in categorical columns via label encoding
    """

    # Identify object/category columns
    object_cols = df.select_dtypes(include=['object', 'category'])
    MAP = []
    # Iterate through each object column
    for col in object_cols:
        # Check if there are NaN values
        if df[col].isnull().values.any():
            # Encode column as numeric with label encoder
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])

            # Get the max category label
            max_label = df[col].max()

            # Set NaN values to max + 1
            df[col] = df[col].fillna(max_label + 1)
            mappings = dict(zip(le.transform(le.classes_), le.classes_))
            MAP.append((col, mappings))
        else:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            mappings = dict(zip(le.transform(le.classes_), le.classes_))
            MAP.append((col, mappings))


    return df, MAP
