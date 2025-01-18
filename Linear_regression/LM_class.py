import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state, test_size, shuffle,
                 fit_intercept, copy_X, n_jobs, positive):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state = random_state # int
        self.test_size = test_size # float
        self.shuffle = shuffle # True/False
        self.fit_intercept = fit_intercept # True/False
        self.copy_X = copy_X # True/False
        self.n_jobs = n_jobs # None/int
        self.positive = positive # True/False


    # Так называемая очистка данных от пустых строк
    def clean_data_and_encoding(self):
        for index, row in self.data.iterrows():
            if row.isnull().sum() > 0:
                self.data.drop(index, axis=0, inplace=True)
        
        # Так называемое горячее кодирование
        categorical_columns = []
        for column in self.data:
            if self.data[column].dtype == 'object':
                categorical_columns.append(column)

        encoder = ce.OneHotEncoder(cols=categorical_columns)
        encoded_data = encoder.fit_transform(self.data)
        self.encoded_data = pd.DataFrame(encoded_data)


    # Так называемая кросс-валидация
    def test_train_validation(self):
        self.X = self.encoded_data.drop([self.target_column], axis = 1)
        self.y = self.data[self.target_column].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, 
                                   random_state=self.random_state,  
                                   test_size=self.test_size,  
                                   shuffle=self.shuffle) 
    
    
    # Так называемое обучение модели
    def train_linear_regression(self):
      
        self.linear_model = LinearRegression(
             fit_intercept=self.fit_intercept, # ключение 'b0' в модель (постоянный член)
             copy_X=self.copy_X, # копирование данных на случай их перезаписи
             n_jobs=self.n_jobs, # количество одновременно задействованных ядер
             positive=self.positive # убирает неположительные коэффициенты из модели
             )
        self.linear_model.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self):
        coefficients = self.linear_model.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.linear_model.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.linear_model.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.linear_model.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.linear_model.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)