import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val, alphas, precompute, eps,
                 tol, warm_start, random_state_lasso, selection, test_size, n_alphas,
                 fit_intercept, copy_X, n_jobs, positive, shuffle, cv, verbose):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int (для валидации)
        self.test_size = test_size
        self.alphas = alphas # list/array-like
        self.n_alphas = n_alphas # int
        self.cv = cv # None/Int
        self.verbose = verbose # True/False
        self.precompute = precompute # True/False/'Auto'
        self.eps = eps # float
        self.tol = tol # float
        self.warm_start = warm_start # True/False
        self.fit_intercept = fit_intercept # True/False
        self.copy_X = copy_X # True/False
        self.n_jobs = n_jobs # None/ int
        self.positive = positive # True/False
        self.random_state_lasso = random_state_lasso # int (Lasso)
        self.selection = selection # 'random'/'cyclic'
        self.shuffle = shuffle # True/False

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
    def train_linear_regression_lasso_cv(self):
      
        self.linear_model_lasso_cv = LassoCV(
             alphas=self.alphas, # коэффициент регуляризации на подбор
             eps = self.eps, # изменение коэффициента на прекращение поиска
             n_alphas=self.n_alphas, # кол-во коэффициентов альфа
             cv=self.cv, # количество повторов перекрестной проверки (None=5)
             fit_intercept=self.fit_intercept, # ключение 'b0' в модель (постоянный член)
             precompute=self.precompute, # нужно ли вычислять матрицу Грамма
             copy_X=self.copy_X, # копирование данных на случай их перезаписи
             n_jobs=self.n_jobs, # указание количества процессоров на выполнение подбора
             tol=self.tol, # порог остановки алгоритма, если изменение функции потень стоновится меньше tol
             verbose=self.verbose, # Детальный вывод отчета по выполнению
             positive=self.positive, # убирает неположительные коэффициенты из модели
             random_state=self.random_state_lasso, # начальное значение генератора случайных числе\
             selection=self.selection # обновлять случайно или циклично коэффициенты на каждой новой интерации?
             )
        self.linear_model_lasso_cv.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self):
        coefficients = self.linear_model_lasso_cv.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.linear_model_lasso_cv.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.linear_model_lasso_cv.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.linear_model_lasso_cv.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.linear_model_lasso_cv.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)