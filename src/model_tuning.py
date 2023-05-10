


numerical_transformer = MinMaxScaler(feature_range=(0,1))

#The best way to encode categorical data is to use CatBoostEncoder
categorical_transformer = Pipeline(steps=[
    ('cat_encoder', ce.CatBoostEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)



final_model = RandomForestRegressor(n_estimators=400, max_features=8, max_depth=16)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)])

final_pipeline.fit(X_train, y_train)
final_preds = final_pipeline.predict(X_test)

model_results = pd.DataFrame([y_test.values, final_preds])
model_results = model_results.transpose()
model_results = model_results.rename(columns={0:'Actual Price',1:'Predicted Price'})

print(model_results.describe(),'\n')
print("RMSE:", round(np.sqrt(mean_squared_error(model_results['Actual Price'], model_results['Predicted Price'])), 1),'\n')
print("MAE:", round(mean_absolute_error(model_results['Actual Price'], model_results['Predicted Price']), 1),'\n')
print("R2 score:", round(r2_score(model_results['Actual Price'], model_results['Predicted Price']), 2))