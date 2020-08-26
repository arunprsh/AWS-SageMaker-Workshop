from numpy import nan
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'
    ],
    target_column_name='y'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as numeric.
    numeric = HEADER.as_feature_indices(
        [
            'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
        ]
    )

    # These features contain a relatively small number of unique items.
    categorical = HEADER.as_feature_indices(
        [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome'
        ]
    )

    numeric_processors = Pipeline(
        steps=[
            (
                'robustimputer',
                RobustImputer(strategy='constant', fill_values=nan)
            )
        ]
    )

    categorical_processors = Pipeline(
        steps=[
            ('thresholdonehotencoder', ThresholdOneHotEncoder(threshold=13))
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric_processing', numeric_processors, numeric
            ), ('categorical_processing', categorical_processors, categorical)
        ]
    )

    return Pipeline(
        steps=[
            ('column_transformer', column_transformer
            ), ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(
        labels=['no'], fill_label_value='yes', include_unseen_class=True
    )
