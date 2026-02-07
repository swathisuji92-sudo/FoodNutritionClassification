# FoodNutritionClassification
This repository uses familiar models of Regression and Classification type to evaluate the Nutritional data in different Food categories

Current Features:

Load Nutritional data(csv - format):
                      Update <Provide user file path> into csv file path
                      Execute cmd - python /FoodNutritionClassification/food_nutrition_analysis.py 

Process done:
  Fetch and display the columns
  Give overview of data
  Normalize the data
  Cut down the outlier data
  Pick the features and target class for model
  Split test and train data
  Feed the data to Models and check its behaviour

Business scenarios covered:
  This trained Model can be used to classify the food or on regression basis it would be able to estimate the target feature value.
  Ex: 
  - Classification models like : LogisticRegression, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, SVC and XGBClassifier can classify if the food is GLUTEN FREE or   not 
  - Regression models like: DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, KNeighborsRegressor and GradientBoostingRegressor can estimate the Calories based on the Food, Preparation, Servings. This helps People who plan the meal, to identify approx Cal.

Future Scope:

  > This can be given a user interface to plan the customer meal

  > Based on intake of calories the meal can be suggested 
