import pandas as pd
import category_encoders as ce
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val, alphas,
                 scoring, cv, gcv_mode, test_size,
                 fit_intercept, store_cv_values, alpha_per_target, shuffle):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int (для валидации)
        self.test_size = test_size
        self.alphas = alphas # list
        self.scoring = scoring # str or None
        self.cv = cv # int or None
        self.fit_intercept = fit_intercept # True/False
        self.store_cv_values = store_cv_values # True/False
        self.alpha_per_target = alpha_per_target # True/False
        self.gcv_mode = gcv_mode # ‘auto’/‘svd’/‘eigen’
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
    def train_linear_regression_ridgecv(self):
      
        self.linear_model_ridgecv = RidgeCV(
            alphas=self.alphas, # коэффициенты на подбор 
            fit_intercept=self.fit_intercept, # ключение 'b0' в модель (постоянный член)
            scoring=self.scoring, # функция оценки модели (по умолчанию r**2)
            cv=self.cv, # Генератор перекрестной проверки или количество заданных инераций кросс-валидации
            gcv_mode=self.gcv_mode, # стратегия перекрестной проверки (auto/svg/eigen)
            store_cv_values=self.store_cv_values, # сохранять результаты перекрестной проверки
            alpha_per_target=self.alpha_per_target, # оптимизировать alpha под каждую цель
            )
        
        self.linear_model_ridgecv.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_coefficients_and_mse(self):
        coefficients = self.linear_model_ridgecv.coef_
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.linear_model_ridgecv.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.linear_model_ridgecv.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.linear_model_ridgecv.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.linear_model_ridgecv.predict(self.X_test))

        print("Коэффициенты модели:")
        print(coefficients)
        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)