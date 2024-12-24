import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data):
        self.data = data

    # Так называемая очистка данных от пустых строк
    def clean_data_and_encoding(self):
        for index, row in self.data.iterrows():
            if row.isnull().sum() > 0:
                self.data.drop(index, axis=0, inplace=True)

        # Так называемое преобразование категориальных переменных
        categorical_columns = []
        for column in self.data:
            if self.data[column].dtype == 'object':
                categorical_columns.append(column)

        encoder = ce.OneHotEncoder(cols=categorical_columns)
        encoded_data = encoder.fit_transform(self.data)
        self.encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names())

    # Так называемая кросс-валидация
    def test_train_validation(self, target_column, random_state, test_size, shuffle):
        X = self.encoded_data.drop([target_column], axis = 1)
        y = self.encoded_data[target_column].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y , 
                                   random_state=random_state,  
                                   test_size=test_size,  
                                   shuffle=shuffle) 
    
    
    # Так называемое обучение модели
    def train_linear_regression(self, fit_intercept, copy_X, n_jobs, positive):
      
        linear_model = LinearRegression(
             fit_intercept=fit_intercept, # ключение 'b0' в модель (постоянный член)
             copy_X=copy_X, # копирование данных на случай их перезаписи
             n_jobs=n_jobs, # количество одновременно задействованных ядер
             positive=positive # убирает неположительные коэффициенты из модели
             )
        linear_model.fit(self.X_train, self.y_train)
        return model
    
    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self, model):
        coefficients = model.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=model.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=model.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=model.predict(self.X_train))
        r_squared_test = r2_score(_true=self.y_test, y_pred=model.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)