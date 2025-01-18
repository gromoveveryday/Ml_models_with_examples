import pandas as pd
import category_encoders as ce
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val, alpha, precompute,
                 tol, warm_start, random_state_lasso, selection, test_size,
                 fit_intercept, copy_X, max_iter, positive, shuffle):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int (для валидации)
        self.test_size = test_size
        self.alpha = alpha # float
        self.precompute = precompute # True/False/'Auto'
        self.tol = tol # float
        self.warm_start = warm_start # True/False
        self.fit_intercept = fit_intercept # True/False
        self.copy_X = copy_X # True/False
        self.max_iter = max_iter # int
        self.positive = positive # True/False
        self.random_state_lasso = random_state_lasso # int (Lasso)
        self.selection = selection # 'random'/'cyclic'
        self.shuffle = shuffle # 


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
                                   random_state=self.random_state_val,  
                                   test_size=self.test_size,  
                                   shuffle=self.shuffle) 
    
    
    # Так называемое обучение модели
    def train_linear_regression_lasso(self):
      
        self.linear_model_lasso = Lasso(
             alpha=self.alpha, # коэффициент регуляризации 
             fit_intercept=self.fit_intercept, # ключение 'b0' в модель (постоянный член)
             precompute=self.precompute, # нужно ли вычислять матрицу Грамма
             copy_X=self.copy_X, # копирование данных на случай их перезаписи
             max_iter=self.max_iter, # максимальное количество итераций
             tol=self.tol, # порог остановки алгоритма, если изменение функции потень стоновится меньше tol
             warm_start=self.warm_start, # Использовать полученные решения предыдущих вызовов при переобучении 
             positive=self.positive, # убирает неположительные коэффициенты из модели
             random_state=self.random_state_lasso, # начальное значение генератора случайных числе
             selection=self.selection # обновлять случайно или циклично коэффициенты на каждой новой интерации?
             )
        self.linear_model_lasso.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self):
        coefficients = self.linear_model_lasso.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.linear_model_lasso.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.linear_model_lasso.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.linear_model_lasso.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.linear_model_lasso.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)