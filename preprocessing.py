import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def drop_num_artifacts(col: pd.Series) -> pd.Series:
    """Extracts numeric characters and decimals from a formatted string series."""
    cleaned_text = col.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(cleaned_text, errors='coerce')


def convert_col_to_binary(col: pd.Series, mapping: dict) -> pd.Series:
    """Maps categorical string values to binary indicators based on a dictionary."""
    return col.map(mapping)


def convert_and_fillna_mode(col: pd.Series, mapping: dict) -> pd.Series:
    """Maps categories to binary values and imputes missing results using the series mode."""
    binary_col = convert_col_to_binary(col, mapping)
    col_mode = binary_col.mode()[0] if not binary_col.mode().empty else 0
    return binary_col.fillna(col_mode)

def fill_median_with_flag(X_train: pd.DataFrame, X_valid: pd.DataFrame, col_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a missingness flag and imputes NaN values using X_train median."""
    X_train_out = X_train.copy()
    X_valid_out = X_valid.copy()

    median_val = X_train_out[col_name].median()

    for df in (X_train_out, X_valid_out):
        df[f'{col_name}_is_missing'] = df[col_name].isnull().astype(int)
        df[col_name] = df[col_name].fillna(median_val)

    return X_train_out, X_valid_out


def convert_with_conditions(df: pd.DataFrame,source_col: str,target_col: str,patterns_dict: dict[str, str],default_val: str = 'Unknown') -> pd.DataFrame:
    """Classifies text into categories based on a dictionary of regex patterns."""

    df_out = df.copy()
    text_series = df_out[source_col].astype(str)

    conditions = [
        text_series.str.contains(pattern, case=False, na=False)
        for pattern in patterns_dict.values()
    ]
    choices = list(patterns_dict.keys())

    df_out[target_col] = np.select(conditions, choices, default=default_val)

    return df_out

def one_hot_encoding(X_train: pd.DataFrame,X_valid: pd.DataFrame,object_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies One-Hot Encoding to categorical features and retains non-categorical columns."""
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    OH_cols_train = pd.DataFrame(
        encoder.fit_transform(X_train[object_cols]),
        columns=encoder.get_feature_names_out(object_cols),
        index=X_train.index
    )

    OH_cols_valid = pd.DataFrame(
        encoder.transform(X_valid[object_cols]),
        columns=encoder.get_feature_names_out(object_cols),
        index=X_valid.index
    )

    num_X_train = X_train.drop(columns=object_cols)
    num_X_valid = X_valid.drop(columns=object_cols)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_valid.columns = OH_X_valid.columns.astype(str)

    return OH_X_train, OH_X_valid



def target_encode_with_smoothing(X_train: pd.DataFrame,X_valid: pd.DataFrame,y_train: pd.Series,col_name: str,min_samples: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replaces high-cardinality categorical values with target means computed on X_train.

    Categories with fewer occurrences than min_samples are flagged as rare and filled
    with the global target mean to prevent overfitting.
    """
    X_train_out = X_train.copy()
    X_valid_out = X_valid.copy()

    global_mean = y_train.mean()
    stats = y_train.groupby(X_train[col_name]).agg(['count', 'mean'])
    reliable_categories = stats[stats['count'] >= min_samples]['mean'].to_dict()

    for df in (X_train_out, X_valid_out):
        encoded = df[col_name].map(reliable_categories)
        df[f'{col_name}_is_rare'] = encoded.isnull().astype(int)
        df[f'{col_name}_encoded'] = encoded.fillna(global_mean)
        df.drop(columns=[col_name], inplace=True)

    return X_train_out, X_valid_out

def process_fuel_type_column(X_train: pd.DataFrame, X_valid: pd.DataFrame, col_name: str = 'fuel_type') -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_out = X_train.copy()
    X_valid_out = X_valid.copy()

    fuel_rules = {
        'Hybrid/Electric': r'Plug-In|Hybrid|Electric|Electri',
        'Flex Fuel': r'Flex Fuel',
        'Diesel': r'Diesel',
        'Gasoline': r'Gasoline|Gas'
    }

    X_train_out = convert_with_conditions(X_train_out, col_name, f'C{col_name}', fuel_rules)
    X_valid_out = convert_with_conditions(X_valid_out, col_name, f'C{col_name}', fuel_rules)

    X_train_out = X_train_out.drop(columns=[col_name])
    X_valid_out = X_valid_out.drop(columns=[col_name])

    X_train_out, X_valid_out = one_hot_encoding(X_train_out, X_valid_out, [f'C{col_name}'])


    return X_train_out, X_valid_out


def process_transmission_column(X_train: pd.DataFrame, X_valid: pd.DataFrame, col_name: str = 'transmission') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts transmission type and speeds count using regex, generates missing indicators,
    and imputes missing values strictly using X_train statistics.
    """
    X_train_out = X_train.copy()
    X_valid_out = X_valid.copy()

    def parse_features(df: pd.DataFrame) -> pd.DataFrame:
        text = df[col_name].astype(str)

        speeds = text.str.extract(r'(\d+)\s*-?\s*Speed', flags=re.IGNORECASE)[0]
        df['num_speeds'] = pd.to_numeric(speeds, errors='coerce')

        conditions = [
            text.str.contains(r'A/T|Automatic|CVT|Dual Shift', case=False, na=False),
            text.str.contains(r'M/T|Manual', case=False, na=False)
        ]
        df[f'{col_name}_type'] = np.select(conditions, ['Automatic', 'Manual'], default='Unknown')
        return df

    X_train_out = parse_features(X_train_out)
    X_valid_out = parse_features(X_valid_out)

    # Missing flags creation before imputation
    for df in (X_train_out, X_valid_out):
        df['num_speeds_is_missing'] = df['num_speeds'].isnull().astype(int)
        df[f'{col_name}_type_is_missing'] = (df[f'{col_name}_type'] == "Unknown").astype(int)

    # Compute training set statistics
    median_speeds = X_train_out['num_speeds'].median()
    if pd.isna(median_speeds):
        median_speeds = 6.0

    known_types = X_train_out[X_train_out[f'{col_name}_type'] != 'Unknown'][f'{col_name}_type']
    mode_transmission = known_types.mode()[0] if not known_types.empty else 'Automatic'

    # Imputation & Binary Encoding
    type_mapping = {"Automatic": 1, "Manual": 0}
    for df in (X_train_out, X_valid_out):
        df['num_speeds'] = df['num_speeds'].fillna(median_speeds)
        df[f'{col_name}_type'] = df[f'{col_name}_type'].replace("Unknown", mode_transmission)
        df[f'{col_name}_type'] = convert_col_to_binary(df[f'{col_name}_type'], type_mapping)
        df.drop(columns=[col_name], inplace=True)

    return X_train_out, X_valid_out

def process_engine_column(X_train: pd.DataFrame, X_valid: pd.DataFrame, col_name: str = 'engine') -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_out = X_train.copy()
    X_valid_out = X_valid.copy()

    def extract_features(df):
        text = df[col_name].astype(str)

        displacement = text.str.extract(r'(\d+(?:\.\d+)?)\s*(?:L|Liter)', flags=re.IGNORECASE)[0]
        df['displacement'] = pd.to_numeric(displacement, errors='coerce')

        hp = text.str.extract(r'(\d+(?:\.\d+)?)\s*HP', flags=re.IGNORECASE)[0]
        df['hp'] = pd.to_numeric(hp, errors='coerce')

        cyl_extracted = text.str.extract(r'(\d+)\s*(?:Cylinder|Cyl)|(?:V|I|Straight\s*)(\d+)', flags=re.IGNORECASE)
        cylinders = cyl_extracted[0].fillna(cyl_extracted[1])
        df['cylinders'] = pd.to_numeric(cylinders, errors='coerce')

        df['is_turbo'] = text.str.contains('Turbo', case=False, na=False).astype(int)

        return df

    X_train_out = extract_features(X_train_out)
    X_valid_out = extract_features(X_valid_out)

    for col in ['displacement', 'hp', 'cylinders']:
        X_train_out, X_valid_out = fill_median_with_flag(X_train_out, X_valid_out, col)

    X_train_out = X_train_out.drop(columns=[col_name])
    X_valid_out = X_valid_out.drop(columns=[col_name])

    return X_train_out,X_valid_out


