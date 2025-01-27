import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Так называемый класс
class DataProcessing:
    def __init__(self, data, target_column, random_state_val, test_size, shuffle,
                 n_estimators, criterion, max_depth, min_samples_split,
                 min_samples_leaf, min_weight_fraction_leaf, max_features,
                 max_leaf_nodes, min_impurity_decrease, bootstrap,
                 oob_score, n_jobs, random_state_mod, verbose,
                 warm_start, ccp_alpha, max_samples):
        self.data = data # объект DataFrame
        self.target_column = target_column # целевой столбец
        self.random_state_val = random_state_val # int
        self.test_size = test_size # float
        self.shuffle = shuffle # True/False
        self.n_estimators = n_estimators # int
        self.criterion = criterion # “squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default=”squared_error”
        self.max_depth = max_depth # int/None
        self.min_samples_split = min_samples_split # int/float
        self.min_samples_leaf = min_samples_leaf # int/float
        self.min_weight_fraction_leaf = min_weight_fraction_leaf # float
        self.max_features = max_features # “sqrt”/ “log2”/ None/ int/float
        self.max_leaf_nodes = max_leaf_nodes # int/None
        self.min_impurity_decrease = min_impurity_decrease # float
        self.bootstrap = bootstrap # True/False
        self.oob_score = oob_score # True/False
        self.n_jobs = n_jobs # int/None
        self.random_state_mod = random_state_mod # int
        self.verbose = verbose # 0/1
        self.warm_start = warm_start # True/False
        self.ccp_alpha = ccp_alpha # float > 0
        self.max_samples = max_samples # int/float


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
    def train_random_forest_regression(self):
      
        self.random_forest_model = RandomForestRegressor(
             n_estimators=self.n_estimators,
             criterion=self.criterion,
             max_depth=self.max_depth,
             min_samples_split=self.min_samples_split, 
             min_samples_leaf=self.min_samples_leaf, 
             min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
             max_features=self.max_features, 
             max_leaf_nodes=self.max_leaf_nodes, 
             min_impurity_decrease=self.min_impurity_decrease, 
             bootstrap=self.bootstrap, 
             oob_score=self.oob_score, 
             n_jobs=self.n_jobs, 
             random_state=self.random_state_mod, 
             verbose=self.verbose, 
             warm_start=self.warm_start, 
             ccp_alpha=self.ccp_alpha, 
             max_samples=self.max_samples
             )
        self.random_forest_model.fit(self.X_train, self.y_train)

    # Так называемый расчет метрик качества регресии
    def evaluate_r2_and_mse(self):
        mse_train = mean_squared_error(y_true=self.y_train, y_pred=self.random_forest_model.predict(self.X_train))
        mse_test = mean_squared_error(y_true=self.y_test, y_pred=self.random_forest_model.predict(self.X_test))
        r_squared_train = r2_score(y_true=self.y_train, y_pred=self.random_forest_model.predict(self.X_train))
        r_squared_test = r2_score(y_true=self.y_test, y_pred=self.random_forest_model.predict(self.X_test))

        print("MSE_train:", mse_train)
        print("MSE_test::", mse_test)
        print("R^2_train:", r_squared_train)
        print("R^2_test:", r_squared_test)