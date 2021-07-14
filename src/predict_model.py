import mlflow

logged_model = 'file:///G:/Meu%20Drive/Machine%20learning%20e%20deep%20learning/Cursos/Alura/Machine%20learning/Machine%20learning%20-%20MLflow%20ciclo%20de%20vidas%20de%20modelos%20ML/Model%20MLfow/src/mlruns/1/5f7aa6ce4a44432b92fd81a21fa26411/artifacts/model'



# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('G:\Meu Drive\Machine learning e deep learning\Cursos\Alura\Machine learning\Machine learning - MLflow ciclo de vidas de modelos ML\Model MLfow\src')
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('precos.csv')