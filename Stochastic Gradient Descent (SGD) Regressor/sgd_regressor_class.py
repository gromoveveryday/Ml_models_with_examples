import pandas as pd
import category_encoders as ce
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val, test_size,
                 shuffle, loss, alpha, penalty, l1_ratio, fit_intercept,
                 max_iter, tol, verbose, random_state_sgd, epsilon,
                 learning_rate, eta0, power_t, early_stopping,
                 validation_fraction, n_iter_no_change, warm_start, average):
        
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int (для валидации)
        self.test_size = test_size
        self.shuffle = shuffle # True/False
        self.loss = loss # squared_error/huber/epsilon_insensitive/squared_epsilon_insensitive
        self.alpha = alpha # float
        self.penalty = penalty # l2/l2/elasticnet/None
        self.l1_ratio = l1_ratio # float
        self.fit_intercept = fit_intercept # True/False
        self.max_iter = max_iter # int
        self.tol = tol # float
        self.verbose = verbose # 0 or 1
        self.random_state_sgd = random_state_sgd # int
        self.epsilon = epsilon # float
        self.learning_rate = learning_rate # invscaling/constant/invscaling/adaptive
        self.eta0 = eta0 # float
        self.power_t = power_t # float
        self.early_stopping = early_stopping # True/False
        self.validation_fraction = validation_fraction # float
        self.n_iter_no_change = n_iter_no_change # int
        self.warm_start = warm_start #True/False
        self.average = average # True/False

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
    def sgd_regressor(self):
      
        self.sgd_regressor = SGDRegressor(
             loss=self.loss, # функция потерь
             alpha=self.alpha, # коэффициент альфа
             penalty=self.penalty,# функция-штраф 
             l1_ratio=self.l1_ratio, # соотношение Lasso и Ridge (0 - 100% Ridge, 1 - 100% Lasso)                            
             fit_intercept=self.fit_intercept, # ключение 'b0' в модель (постоянный член)
             max_iter=self.max_iter, # максимальное количество итераций
             tol=self.tol, # порог остановки алгоритма, если изменение функции потень стоновится меньше tol
             verbose=self.verbose, # Уровень детализации выводимой информации обучения от 0 до 1
             shuffle=self.shuffle, # перемешивать ли данные после каждой итерации
             random_state=self.random_state_sgd, # начальное значение генератора случайных числе
             epsilon=self.epsilon, # значение эпсилон в функции потерь
             learning_rate=self.learning_rate, # темп обучения
             eta0=self.eta0, # Начальная скорость обучения
             power_t=self.power_t, # значение экспоненты для обратного сглаживания скорости обучения
             early_stopping=self.early_stopping, # включить раннюю установка при отсутсвии эффекта
             validation_fraction=self.validation_fraction, # доля проверочной выборки для ранней остановки
             n_iter_no_change=self.n_iter_no_change, # количество безрезультатных итераций для остановки
             warm_start=self.warm_start, # использовать прошлые результаты при повторном обучении
             average=self.average, # расчет усрелненных для всех значений SGD
             )
        self.sgd_regressor.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self):
        coefficients = self.sgd_regressor.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.sgd_regressor.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.sgd_regressor.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.sgd_regressor.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.sgd_regressor.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)