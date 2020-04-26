import os
import pickle
import joblib
import time
import mlflow.pyfunc
from src.validate import validate, cross_validate


# Define the model class
class Model(mlflow.pyfunc.PythonModel):
    """
    Model Class: it can be initializate with an instance of any model class
                        or with a model path to an untrained or trained pickle.
                        The Class that represents the model has to have a FIT and PREDICT method.
        Fit method:     includes the preprocess logic to train a model
        Predict Method: includes the preprocess and postprocess logic
    """
    def __init__(self, model):
        self._model = model
        
    def fit(self, train_x, train_y):
        """
        Load raw data, preprocess it and fit the model to the train data
        Record preprocess + train elapsed time
        """
        # Guardamos train size y train features
        [self.train_size, self.features] = train_x.shape
        print('{:=^80}'.format('  TRAINING MODEL  '))
        print("Starting training...")
        t0 = time.time()
        #train_x = self._preprocessor.preprocess(train_x)
        self._model.fit(train_x, train_y)
        self.train_time = time.time() - t0
        print("Model trained in {} s.".format(self.train_time))
        print('{:=^80}'.format(''))

    def predict(self, model_input):
        """
        Clean data and infer data (predict)
        """
        model_input = clean_data(model_input)
        predicted_target = self._infer(model_input)
        predicted_target = self._postprocessor.postprocess(predicted_target)
        return predicted_target

    def _infer(self, model_input):
        """
        Infer data and measure infer time.
        """
        t0 = time.time()
        predicted_target = self._model.predict(model_input)
        self.inference_time = time.time() - t0
        return predicted_target

    def evaluate(self, test_x, test_y):
        """
        Takes two numpy arrays test_x, test_y:
        returns a dictionary with the metrics
        to evaluate a productive model (with comebacks)
        and also the training time + inference time
        """
        self.test_size = test_x.shape[0]
        print('{:=^80}'.format('  VALIDATE MODEL  '))
        print("Evaluating productive model...")
        pred_y = self._infer(test_x)
        # Validamos las predicciones de test
        metrics = validate(test_y, pred_y)
        # Incluimos metricas de training
        metrics['train_time'] = self.train_time
        metrics['mean_inference_time'] = self.inference_time/self.test_size
        metrics['train_size'] = self.train_size
        metrics['test_size'] = self.test_size
        print('{:=^80}'.format(''))
        return metrics
    
    def evaluate_predictive(self, test_x, test_y):
        """
        Takes two numpy arrays test_x, test_y:
        returns a dictionary with the metrics
        to evaluate a predictive model (without comebacks)
        and also the training time + inference time
        """
        print("=== Evaluating predictive model...")
        #test_x = self._preprocessor.preprocess(test_x) 
        pred = self._model.predict(test_x)
        metrics = validate(test_y, pred)
        metrics['train_time'] = self.train_time
        return metrics

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        return cls(model)

if __name__ == "__main__":
    print("========= Init MLFlow Custom Model TEST with KNN =========")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    model = CustomModel(knn)
    model.fit()
    model.evaluate()
    model.evaluate_predictive()