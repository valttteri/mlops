import pickle
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import pandas as pd

# import great_expectations as gx
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder

from thefuzz import fuzz, process

# from great_expectations.data_context import FileDataContext
# from great_expectations.datasource.fluent import BatchRequest
# from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


# Random seed for making the assignments reproducible
RANDOM_SEED = 42


def file_reader(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read files of deals and house information into DataFrames
    Args:
        path (Path): Path to the folder where the files exist.
    Returns:
        a tuple consisting of a Pandas DataFrame of deals and a Pandas DataFrame of house information
    """

    price_file_path = path / "deals.csv"
    house_file_path = path / "house_info.json"

    ### BEGIN SOLUTION
    price_data = pd.read_csv(price_file_path, index_col=None)
    house_data = pd.read_json(house_file_path, orient="records")

    return (price_data, house_data)
    ### END SOLUTION


def dataframe_merger(prices: pd.DataFrame, house_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the two DataFrames given as inputs
    Args:
        prices (DataFrame): A pandas DataFrame holding the price information
        house_data (DataFrame): A pandas DataFrame holding the detailed information about the sold houses
    Returns:
        The merged pandas DataFrame
    """

    ### BEGIN SOLUTION
    prices["date"] = pd.to_datetime(prices["datesold"], infer_datetime_format=True)
    prices_renamed = prices.rename(columns={"building_year": "yr_built"}, inplace=False)
    df = pd.merge(
        prices_renamed,
        house_data,
        on=["date", "postcode", "bedrooms", "area", "yr_built"],
    )
    return df
    ### END SOLUTION


def drop_futile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unneeded columns from the argument DataFrame
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
    Returns:
        The pandas DataFrame with the columns listed in the instruction removed
    """

    ### BEGIN SOLUTION
    return df.drop(
        columns=["yr_renovated", "prev_owner", "datesold", "sqft_above"], inplace=False
    )
    ### END SOLUTION


def correct_distance_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct the falsely input values in column 'distance'
    Args:
        df (DataFrame): A Pandas DataFrame holding all of the housing data
    Returns:
        The Pandas DataFrame with the values in column 'distance' corrected
    """
    df_copy = df.copy()

    ### BEGIN SOLUTION
    mask = df_copy["distance"] >= 100
    df_copy.loc[mask, "distance"] = df_copy.loc[mask, "distance"] / 1000
    return df_copy
    ### END SOLUTION


def string_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercases all values in the column 'condition' and removes trailing white space.
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
    Returns:
        The pandas DataFrame with the 'condition' column values being lower-cased and stripped of trailing white space
    """
    df_copy = df.copy()
    ### BEGIN SOLUTION
    df_copy["condition"] = df_copy["condition"].str.lower()
    df_copy["condition"] = df_copy["condition"].str.strip()
    return df_copy
    ### END SOLUTION


def typo_fixer(
    df: pd.DataFrame, threshold: float, correct_condition_values: List[str]
) -> pd.DataFrame:
    """
    Uses fuzzy string matching to fix typos in the column 'condition'. It loops through each entry in the column and
    replaces them with suggested corrections if the similarity score is high enough.
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data
        threshold (int): A number between 0-100. Only the entries with score above this number are replaced.
        correct_condition_values (List): A list of correct strings that we hope the condition column to include. For example, correct_ones=['excellent', 'good', 'satisfactory'] in the case of the training dataset.
    Returns:
        The pandas DataFrame with the 'condition' column values corrected
    """
    df_copy = df.copy()

    ### BEGIN SOLUTION
    matches = []
    scores = []

    for entry in df_copy["condition"]:
        best_match, score = process.extractOne(
            entry, correct_condition_values, scorer=fuzz.ratio
        )

        if score > threshold:
            matches.append(best_match)
        else:
            matches.append(entry)
        scores.append(score)
    df_copy["condition"] = matches
    df_copy["similarity_scores"] = scores
    return df_copy
    ### END SOLUTION


def data_extraction(path: Path, correct_condition_values: List[str]) -> pd.DataFrame:
    """
    The entire data extraction/cleaning pipeline wrapped inside a single function.
    Args:
        path (Path): Path to the folder where the files exist.
        correct_condition_values (List): A list of correct strings that we hope the condition column to include.
    Returns:
        df (DataFrame): A pandas DataFrame holding all of the (cleaned) housing data.
    """
    threshold = 80
    prices, house_data = file_reader(path)
    df_merged = dataframe_merger(prices, house_data)
    df_with_columns_dropped = drop_futile_columns(df_merged)
    df_with_corrected_distance_unit = correct_distance_unit(df_with_columns_dropped)
    df_with_condition_column_transformed = string_transformer(
        df_with_corrected_distance_unit
    )
    df_typo_fixed = typo_fixer(
        df_with_condition_column_transformed, threshold, correct_condition_values
    )
    return df_typo_fixed


# def batch_creator(df: pd.DataFrame, context: FileDataContext, data_source_name: str) -> BatchRequest:
#     """
#     Creates a new Batch Request using the given DataFrame
#     Args:
#         df (DataFrame): A pandas DataFrame holding the cleaned housing data.
#         context (GX FileDataContext): The current active GX Data Context
#         data_source_name (str): Name of the GX Data Source, to which the DataFrame is added
#     Returns:
#         new_batch_request (GX BatchRequest): The GX batch request created using df
#     """
#     datasource = context.get_datasource(data_source_name)
#     test_asset_name = "test_data"

#     ### BEGIN SOLUTION
#     try:
#         # Create a new DataFrame asset
#         test_data_asset = datasource.add_dataframe_asset(name=test_asset_name)
#     except ValueError:
#         # The asset already exists
#         test_data_asset = datasource.get_asset(test_asset_name)

#     new_batch_request = test_data_asset.build_batch_request(dataframe=df)

#     return new_batch_request
#     ### END SOLUTION


# def create_checkpoint(context: FileDataContext, batch_request: BatchRequest, checkpoint_name: str, expectation_suite_name: str, run_name: str) -> CheckpointResult:
#     """
#     Creates a new GX Checkpoint from the argument batch_request
#     Args:
#         context (GX FileDataContext): The current active context
#         new_batch_request (GX BatchRequest): A GX batch request used to create the Checkpoint
#         checkpoint_name (str): Name of the Checkpoint
#         expectation_suite_name (str): Name of the Expectation Suite used to validate the data
#         run_name (str): Name of the validation running
#     Returns:
#         checkpoint_result (GX CheckpointResult):
#     """

#     ### BEGIN SOLUTION
#     checkpoint = context.add_or_update_checkpoint(
#         name=checkpoint_name,
#         batch_request=batch_request,
#         expectation_suite_name=expectation_suite_name
#     )
#     checkpoint_result = checkpoint.run(run_name=run_name)

#     return checkpoint_result
#     ### END SOLUTION


def separate_X_and_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the features and targets.
    Args:
        df (DataFrame): A pandas DataFrame holding the cleaned housing data
        target (str): Name of the target column
    Returns:
        X (DataFrame): A Pandas DataFrame holding the cleaned housing data without the target column
        y (Series): A pandas Series with the target values
    """
    y = df[target].astype("float64")
    X = df.loc[:, df.columns != target]
    return (X, y)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing numerical values using MICE.
    Args:
        df (DataFrame): A pandas DataFrame holding the features
    Returns:
        The pandas DataFrame with the missing values imputed
    """
    imp = IterativeImputer(random_state=RANDOM_SEED, add_indicator=True)

    # A list of column names where missing values need to be imputed.
    # We'll not include categorical or datetime variables in the calculations
    included_columns = [
        True if x not in ["postcode", "area", "date", "condition"] else False
        for x in df.columns
    ]

    df_copy = df.copy()
    ### BEGIN SOLUTION
    _columns = df_copy.loc[:, included_columns]
    imputed_values = imp.fit_transform(_columns)

    column_names = imp.get_feature_names_out()
    df_copy.loc[:, column_names] = imputed_values
    return df_copy
    ### END SOLUTION


def datetime_decomposer(df: pd.DataFrame, dt_column_name: str) -> pd.DataFrame:
    """
    Decomposes datetime values into year, quarter, and weekday.
    Args:
        df (DataFrame): A pandas DataFrame holding the features
        dt_column_name(str): The name of the datetime column
    Returns:
        The pandas DataFrame with the datetime column decomposed
    """
    df_copy = df.copy()
    ### BEGIN SOLUTION
    df_copy["year"] = df_copy[dt_column_name].dt.year
    df_copy["quarter"] = df_copy[dt_column_name].dt.quarter
    df_copy["weekday"] = df_copy[dt_column_name].dt.dayofweek
    return df_copy.drop(columns=[dt_column_name], inplace=False)
    ### END SOLUTION


def condition_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes conditions to numerical range 1-5
    Args:
        df (DataFrame): A pandas DataFrame holding the features
    Returns:
        The pandas DataFrame with the condition column encoded
    """
    df_copy = df.copy()
    ### BEGIN SOLUTION
    condition_map = {
        "poor": 1,
        "tolerable": 2,
        "satisfactory": 3,
        "good": 4,
        "excellent": 5,
    }
    df_copy["condition"] = df_copy["condition"].map(condition_map)
    return df_copy
    ### END SOLUTION


def target_encode(
    df: pd.DataFrame,
    columns: List[str],
    target: Optional[pd.Series] = None,
    encoder: Optional[TargetEncoder] = None,
) -> Tuple[TargetEncoder, pd.DataFrame]:
    """
    Encodes postcode and area to numerical format using a target encoder
    Args:
        df (DataFrame): A pandas DataFrame holding the features
        columns (list of strings): Names of the categorical columns to be encoded
        target (Series|None): A pandas Series with the target values. This is required only when fitting the encoder.
        encoder(TargetEncoder|None): An already fitted encoder. This is required when we want to apply an encoder, which has already been fitted during training.
    Returns:
        A tuple consisting of the fitted encoder (either a new fitted one or the one passed as an argument) and the pandas DataFrame with the categorical columns encoded
    """
    if encoder is None:
        encoder = TargetEncoder(
            target_type="continuous", smooth="auto", random_state=RANDOM_SEED
        )

    df_copy = df.copy()

    ### BEGIN SOLUTION
    if target is not None:
        encoder.fit(df_copy[columns], target)
    df_copy[columns] = encoder.transform(df_copy[columns])
    ### END SOLUTION

    df_copy[columns] = df_copy.loc[:, columns].astype("float64")

    return encoder, df_copy


def store_features(X: pd.DataFrame, feature_file_path: str) -> None:
    """
    Stores a set of features to a specified location
    Args:
        X (DataFrame): A pandas DataFrame holding the features
        feature_file_path (Path): Path for the stored features, e.g., feature_store/housing_train_X.parquet
    """

    ### BEGIN SOLUTION
    X.to_parquet(feature_file_path)
    ### END SOLUTION


def store_targets(y: pd.Series, target_file_path: Path) -> None:
    """
    Stores a set of features to a specified location
    Args:
        y (Series): A pandas Series holding the target values
        target_file_path (Path): Path for the stored targets, e.g., feature_store/housing_train_y.csv
    """

    ### BEGIN SOLUTION
    y.to_csv(target_file_path, index=False)
    ### END SOLUTION


def store_encoder(encoder: TargetEncoder, encoder_file_path: Path) -> None:
    """
    Stores a targetEncoder to a specified location
    Args:
        encoder (TargetEncoder): A fitted scikit-learn TargetEncoder object
        encoder_file_path (Path): Path of the stored target encoder.
    """

    ### BEGIN SOLUTION
    with open(encoder_file_path, "wb") as enc_f:
        pickle.dump(encoder, enc_f)
    ### END SOLUTION


def drop_unimportant_features(df: pd.DataFrame):
    """
    df (DataFrame): DataFrame from which the unimportant features should be dropped.
    """
    unimportant_features = [
        "floors",
        "similarity_scores",
        "missingindicator_sqft_living15",
        "missingindicator_sqft_lot15",
        "quarter",
        "weekday",
    ]
    ### BEGIN SOLUTION
    return df.drop(columns=unimportant_features, inplace=False, errors="ignore")
    ### END SOLUTION


def feature_engineering_pipeline(
    df: pd.DataFrame,
    feature_store_path: Path,
    feature_file_name: str,
    encoder_file_name: str,
    target_file_name: Optional[str] = None,
    fit_encoder: bool = False,
    targets_included: bool = True,
) -> None:
    """
    Converts a given (merged) housing data DataFrame into features and targets, performs feature engineering, and
    stores the features along with possible targets and a fitted encoder
    Args:
        df (DataFrame): A pandas DataFrame holding all of the housing data, or just the features (see targets_included)
        feature_store_path (Path): Path of the feature store
        feature_file_name (str): Filename for the stored features.
        encoder_file_name (str): Filename for the stored encoder.
        target_file_name (str|None): Filename for the stored targets.
        fit_encoder (bool): Whether a new target encoder should be fitted. If False, uses a previously stored encoder
        targets_included (bool):  If True, df has all of the housing data including targets. If False, df has only the features.
    """
    if targets_included:
        X, y = separate_X_and_y(df, target="price")
    else:
        if fit_encoder:
            raise ValueError("Target encoder can not be trained without targets.")

    X_missing_imputed_ = impute_missing(X)
    X_datetime_decomposed = datetime_decomposer(
        X_missing_imputed_, dt_column_name="date"
    )
    X_condition_encoded = condition_encoder(X_datetime_decomposed)
    X_unimportant_features_dropped = drop_unimportant_features(X_condition_encoded)

    feature_file_path = feature_store_path / feature_file_name
    target_file_path = feature_store_path / target_file_name
    encoder_file_path = feature_store_path / "encoders" / encoder_file_name

    if fit_encoder:
        t_encoder, X_target_encoded = target_encode(
            X_unimportant_features_dropped, columns=["postcode", "area"], target=y
        )
        store_features(X_target_encoded, feature_file_path)
        store_targets(y, target_file_path)
        store_encoder(t_encoder, encoder_file_path)

    else:
        with open(encoder_file_path, "rb") as enc_f:
            t_encoder = pickle.load(enc_f)
        _, X_target_encoded = target_encode(
            X_unimportant_features_dropped,
            columns=["postcode", "area"],
            encoder=t_encoder,
        )
        store_features(X_target_encoded, feature_file_path)
        if targets_included:
            store_targets(y, target_file_path)


def etl(
    path: Path,
    # gx_context_root_dir: Path,
    # gx_datasource_name: str,
    # gx_checkpoint_name: str,
    # gx_expectation_suite_name: str,
    # gx_run_name: str,
    feature_store_path: Path,
    feature_file_name: str,
    encoder_file_name: str,
    target_file_name: Optional[str] = None,
    fit_encoder: bool = False,
    targets_included: bool = True,
):
    """
    This function loads, merges, cleans, and validates the specified data, extract features, and save the features in the feature store.
    Args:
        path (Path): Path to the folder where the files "deals.csv" and "housing_info.json" exist
        feature_store_path (Path): Path of the feature store
        feature_file_name (str): Filename for the stored features
        encoder_file_name (str): Filename for the stored target encoder
        target_file_name (None|str): Filename for the stored targets
        fit_encoder (bool): Whether a new target encoder should be fitted. If False, uses a previously stored encoder
        targets_included (bool): If True, df has all of the housing data including targets. If False, df has only the features.
    """
    df = data_extraction(
        path,
        correct_condition_values=[
            "poor",
            "tolerable",
            "satisfactory",
            "good",
            "excellent",
        ],
    )
    # context = gx.get_context(context_root_dir=gx_context_root_dir)
    # new_batch_request = batch_creator(df=df, context=context, data_source_name=gx_datasource_name)
    # checkpoint_result = create_checkpoint(
    #     context=context,
    #     batch_request=new_batch_request,
    #     checkpoint_name=gx_checkpoint_name,
    #     expectation_suite_name=gx_expectation_suite_name,
    #     run_name=gx_run_name
    # )
    # if (not checkpoint_result.success):
    #     logging.warning("Some GX validations failed")

    feature_engineering_pipeline(
        df=df,
        feature_store_path=feature_store_path,
        feature_file_name=feature_file_name,
        encoder_file_name=encoder_file_name,
        target_file_name=target_file_name,
        fit_encoder=fit_encoder,
        targets_included=targets_included,
    )
