import pandas as pd
import category_encoders as ce
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val,
                 test_size, shuffle, criterion, splitter, max_depth,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_features, random_state, min_impurity_decrease,
                 ccp_alpha, max_leaf_nodes):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int (для валидации)
        self.test_size = test_size
        self.shuffle = shuffle # True/False
        self.criterion = criterion # “squared_error”, “friedman_mse”, “absolute_error”, “poisson”
        self.splitter = splitter # “best”, “random”
        self.max_depth = max_depth # int/None
        self.min_samples_split = min_samples_split # int/float/2
        self.min_samples_leaf = min_samples_leaf # int/float/1
        self.min_weight_fraction_leaf = min_weight_fraction_leaf # float/0
        self.max_features = max_features # int/float/“sqrt”, “log2”/None
        self.random_state = random_state # int/None
        self.min_impurity_decrease = min_impurity_decrease # float/0
        self.ccp_alpha = ccp_alpha # >= 0 float/0
        self.max_leaf_nodes = max_leaf_nodes # int/None

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
    def decision_tree_reg_model(self):
      
        self.decision_tree_reg_model = DecisionTreeRegressor(
            criterion=self.criterion, # функция определения качества разделения
            splitter=self.splitter, # Стратегия разделения на каждом узле
            max_depth=self.max_depth, # Максимальная глубина дерева
            min_samples_split=self.min_samples_split, # достаточное количество выборок для разделения узла
            min_samples_leaf=self.min_samples_leaf, # Минимальное количество выборок чтобы находиться в конечном узле
            min_weight_fraction_leaf=self.min_weight_fraction_leaf, # минимальная взвешенная доля от общей суммы весов
            max_features=self.max_features, # Количество функций учитывающий при оценки качества разделения
            random_state=self.random_state, # мера управления случайноости оценки
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha, # Параметр альфа
            max_leaf_nodes=self.max_leaf_nodes
            )
        
        self.decision_tree_reg_model.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_mse(self):
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.decision_tree_reg_model.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.decision_tree_reg_model.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.decision_tree_reg_model.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.decision_tree_reg_model.predict(self.X_test))

        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)