from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        self.dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        self.time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.preproc_pipe = ColumnTransformer([
            ('distance', self.dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', self.time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipe = Pipeline([
            ('preproc', self.preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        self.trained_pipe = self.pipe.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.trained_pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    data = get_data()
    cleaned_data = clean_data(data)
    X = cleaned_data.drop("fare_amount", axis=1)
    y = cleaned_data["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    train_object = Trainer(X, y)
    train_object.set_pipeline()
    train_object.run()
    rmse = train_object.evaluate(X_test, y_test)
    print(rmse)
    print('TODO')

print("done")
