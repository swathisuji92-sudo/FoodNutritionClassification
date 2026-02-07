import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

terminator="_"*100

def outlier(filtered_outlier_range_df,valid_data_df):
    print("Features that are Outliers:")
    print(filtered_outlier_range_df)
    for i in filtered_outlier_range_df['Column'].items():
        outlier_col=i[1]
        print(outlier_col)
        plt.figure(figsize=(8, 6))
        plt.boxplot(valid_data_df[outlier_col], vert=True, patch_artist=True,whis=1.5)
        plt.title(f'Box Plot for the "{outlier_col}" Column')
        plt.ylabel('Value Range')
        plt.xticks([1], [f'{outlier_col}']) 
        plt.grid(False)

        # Display the plot
        plt.show()
    
def serving_vs_outlier_visualization(filtered_outlier_range_df,valid_data_df):
    for i in filtered_outlier_range_df.iterrows():
        outlier_col=i[1]
        col_name=outlier_col['Column']
        print(col_name,outlier_col['Upper'])
        high_rec=valid_data_df[valid_data_df[col_name]>outlier_col['Upper']]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=high_rec, x=f'{col_name}', y='Serving_Size', hue='Food_Name')
        plt.title(f"{col_name} vs Serving Size ")
        plt.show()
        
def regression_model_test(model, X_train, y_train):
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    score=r2_score(y_test, y_pred)
    # Metrics
    metric = {
        "Model": name,
        "MSE": round(mean_squared_error(y_test, y_pred),2),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": score,
        "R2_Train":r2_score(y_train,y_train_pred),
        "R2_Test": r2_score(y_test,y_test_pred)
    } 
    return metric,score

def classification_model_test(model, X_train, y_train):
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    classification_metric = {
        "Model": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "accuracy": acc,
    }
    return acc,y_pred,classification_metric

df=pd.read_csv('<Provide user file path>')
print(df.info())
print(df.head(5))
# updated_df=df.replace(r'^\s*$', np.nan, regex=True)
# print(updated_df[updated_df['Calories'].isna()])
mask_all_missing = df.isna().any(axis=1)
rows_with_missing_data = df[mask_all_missing]

null_val_columns = df.columns[df.isna().any()].tolist()
print(terminator)
print("Columns with null values::",null_val_columns)

clean_data = df.dropna()
print(terminator)
print("Clean data after dropping NA: \n",clean_data.shape)

clean_data.drop_duplicates()
print(terminator)
print("Clean data after dropping duplicates: \n",clean_data.shape)

print(terminator)
print("Missing Data:")
print(rows_with_missing_data.head(5))
print(rows_with_missing_data.shape)
print(rows_with_missing_data[["Food_Name","Meal_Type","Preparation_Method","Is_Vegan","Is_Gluten_Free"]].value_counts())

#As there are duplicates in missing values, am removing duplicate rows(Having it might be meaningless)
rows_with_missing_data=rows_with_missing_data.drop_duplicates()
print(terminator)
print("Missing Data after dropping duplicates:")
print(rows_with_missing_data[["Food_Name","Meal_Type","Preparation_Method","Is_Vegan","Is_Gluten_Free"]].value_counts())
print(rows_with_missing_data.shape)


for i in null_val_columns:
    #print(i)
    rows_with_missing_data[i]=rows_with_missing_data.apply(lambda row: np.mean(clean_data[(clean_data['Food_Name'] == row['Food_Name']) & (clean_data['Is_Gluten_Free'] == row['Is_Gluten_Free']) 
                                                                & (clean_data['Is_Vegan'] == row['Is_Vegan']) & (clean_data['Preparation_Method'] == row['Preparation_Method']) 
                                                                & (clean_data['Meal_Type'] == row['Meal_Type'])][i]),axis=1)

print(terminator)
print("rows_with_missing_data DF after NULL handling:")
print(rows_with_missing_data.head(5))

check_missin_data=rows_with_missing_data[rows_with_missing_data.isna().any(axis=1)]
print(terminator)
print("Check if missing data after handling:",check_missin_data.shape)

#Merge the Clean Data and Handled missing data DF together to make Valid Data DF

valid_data_df=pd.concat([clean_data,rows_with_missing_data],ignore_index=True)
print(terminator)
print(f"Merged the clean data with interpreted data for Null values: \n {clean_data.shape[0]} + {rows_with_missing_data.shape[0]} = {clean_data.shape[0] + rows_with_missing_data.shape[0]}")
print(valid_data_df.info())
print(valid_data_df.head())

#Convert all numerical columns in same measure (g), as it avoids over weightage to Sodium and Cholestrol as those are measured in (mg)
print(terminator)
print('Convert Sodium and Cholestrol into grams:')
valid_data_df['Sodium']=valid_data_df.apply(lambda row : row['Sodium']/1000, axis=1)
valid_data_df['Cholesterol']=valid_data_df.apply(lambda row : row['Cholesterol']/1000, axis=1)

print(valid_data_df.head())


#Class imbalance
print(terminator)
print('Categorical columns:')
categorical_cols=df.select_dtypes(include=['object', 'bool']).columns
print(categorical_cols)

for col in categorical_cols:
    print(terminator)
    print(f"Data Distribution for the class : {col}")
    val_cnt=valid_data_df[col].value_counts(normalize=True)
    print(val_cnt)

    labels = val_cnt.index
    sizes  = val_cnt.values

    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'{col} - Class Proportions')
    plt.tight_layout()
    plt.show()
print(terminator)

#Get Min and Max of each numerical column to understand outlier pattern
numerical_cols=df.select_dtypes(include=['number']).columns
print(numerical_cols)

outlier_range_df=pd.DataFrame(columns=['Column','Lower','Upper','Min','Max','No of Low Records','No of High Records'])

for col in numerical_cols:
    #print(terminator)
    #print(f"Data range for the column : {col}")
    val_cnt=valid_data_df[col].value_counts(normalize=True)
    min=valid_data_df[col].min()
    max=valid_data_df[col].max()
    #print("Min:",min,"\nMax:",max)

    #IQR used as data is extremly varying in Calorie and Sodium columns
    Q1 = valid_data_df[col].quantile(0.25)
    Q3 = valid_data_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    no_of_low_rec=valid_data_df[valid_data_df[col]<lower].shape[0]
    no_of_high_rec=valid_data_df[valid_data_df[col]>(upper)].shape[0]
    outlier_range_df.loc[len(outlier_range_df)] = [col,lower,upper,min,max,no_of_low_rec,no_of_high_rec]
outlier_range_df['is_oulier'] = outlier_range_df.apply(lambda r: 1 if ((r['Min']-r['Lower'] >50 or r['Max']-r['Upper']>50) and (r['Max'] - r['Min'] > 100)) else 0, axis=1)
print(outlier_range_df)
print(terminator)

filtered_outlier_range_df=outlier_range_df[outlier_range_df['is_oulier'] == 1]
outlier(filtered_outlier_range_df,valid_data_df)
filtered_outlier_range_df=filtered_outlier_range_df[filtered_outlier_range_df['Column']!='Serving_Size']
serving_vs_outlier_visualization(filtered_outlier_range_df,valid_data_df)



print(valid_data_df[valid_data_df['Serving_Size']>500].shape[0])
print(valid_data_df[valid_data_df['Serving_Size']>500].head(100))

filtered_df=valid_data_df[valid_data_df['Serving_Size']<500]
outlier(filtered_outlier_range_df,filtered_df)
serving_vs_outlier_visualization(filtered_outlier_range_df,filtered_df)

encoded_df=filtered_df
encoded_df['Is_Vegan']=encoded_df.apply(lambda row: 1 if row['Is_Vegan']==True else 0,axis=1)
encoded_df['Is_Gluten_Free']=encoded_df.apply(lambda row: 1 if row['Is_Gluten_Free']==True else 0,axis=1)

# Correlation
numerical_cols=encoded_df.select_dtypes(include=['number']).columns
print(numerical_cols)
corr_df=encoded_df[numerical_cols].corr()
print(corr_df)
plt.figure(figsize=(10, 10))
sns.heatmap(data=corr_df, annot=True, cmap='coolwarm')
plt.title("Correlation Map")
plt.show()

#OneHot encoding for Meal_Type
encoded_df=pd.get_dummies(filtered_df, columns=['Meal_Type'],prefix=['Meal'],dtype=int)
print(encoded_df.head())

oe=OrdinalEncoder(categories=[['raw','baked','grilled','fried']])
encoded_df['Encoded_Preparation_Method']=oe.fit_transform(encoded_df[['Preparation_Method']])
# le=LabelEncoder()
# encoded_df['Encoded_Preparation_Method']=le.fit_transform(encoded_df[['Preparation_Method']]) # Not performing good for this data
print(encoded_df[['Preparation_Method','Encoded_Preparation_Method']])

oe=OrdinalEncoder(categories=[['Apple','Banana','Salad','Sushi','Pasta','Steak','Ice Cream', 'Donut', 'Burger', 'Pizza']])
encoded_df['Encoded_Food_Name']=oe.fit_transform(encoded_df[['Food_Name']])
# le=LabelEncoder()
# encoded_df['Encoded_Food_Name']=le.fit_transform(encoded_df[['Food_Name']]) # Not performing good for this data
print(encoded_df[['Food_Name','Encoded_Food_Name']])

# Regression
print(terminator)
print("-----------------Regression-----------------")
print(terminator)
features = ["Protein", "Fat", "Carbs", "Cholesterol","Sodium", "Serving_Size", "Encoded_Food_Name","Fiber","Sugar"]
regression_target = "Calories"
X = encoded_df[features]
y = encoded_df[regression_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg_models = {
    "DecisionTreeRegression",
    "RandomForest",
    "KNearestNeighbour",
    #"SupportVectorRegression",# Not working
    "XGBRegressor",
    "GradientBoostingRegressor"
}
metrics = []
feature_importance = {}

for name in reg_models:
    print(terminator)
    print(name)
    print(terminator)
    old_score=0
    reg_model_metric={}
    if name == "DecisionTreeRegression":
        for i in range(1,10):
            model = DecisionTreeRegressor(criterion='squared_error', max_depth=i, random_state=42)
            metric,score = regression_model_test(model, X_train, y_train)
            print("prev score::",old_score,":current score::",score)
            reg_model_metric[i]=score
            if score <= old_score:
                model = DecisionTreeRegressor(criterion='squared_error', max_depth=i-1, random_state=42)
                metric,score = regression_model_test(model, X_train, y_train)
                break
            old_score=score
    elif name == "RandomForest":

        for i in range(10,150,10):
            model=RandomForestRegressor(n_estimators=i, random_state=42)            
            metric,score = regression_model_test(model, X_train, y_train)
            print("prev score::",old_score,":current score::",score)
            reg_model_metric[i]=score
            if score <= old_score:
                model=RandomForestRegressor(n_estimators=i-10, random_state=42)
                metric,score = regression_model_test(model, X_train, y_train)
                break
            old_score=score
    elif (name == "KNearestNeighbour" or name == "XGBRegressor"):
        for i in range(1,50,5):
            model=KNeighborsRegressor(n_neighbors=i) 
            if name == "XGBRegressor":
                model = xgb.XGBRegressor(n_estimators=i, learning_rate=0.1)          
            metric,score = regression_model_test(model, X_train, y_train)
            print("prev score::",old_score,":current score::",score)
            reg_model_metric[i]=score
            if score <= old_score:
                model=KNeighborsRegressor(n_neighbors=i-5)
                if name == "XGBRegressor":
                    model = xgb.XGBRegressor(n_estimators=i-5, learning_rate=0.1)  
                metric,score = regression_model_test(model, X_train, y_train)
                break
            old_score=score
    elif name == "GradientBoostingRegressor":
        final_depth=5
        break_flag=False
        for i in range(1,100,5):
            if break_flag:
                break
            grad_model_metric={}
            for depth in range(3,6):
                model=GradientBoostingRegressor(n_estimators=i,learning_rate=0.1,max_depth=depth,random_state=42)        
                metric,score = regression_model_test(model, X_train, y_train)
                print("In n_estimator::",i,"depth::",depth,"prev score::",old_score,":current score::",score)
                grad_model_metric[depth]=score
                if score <= old_score:
                    break_flag=True
                    final_depth=depth
                    break
                old_score=score
        
        old_score=0

        for i in range(1,100,5):
            model=GradientBoostingRegressor(n_estimators=i,learning_rate=0.1,max_depth=final_depth,random_state=42)        
            metric,score = regression_model_test(model, X_train, y_train)
            reg_model_metric[i]=score
            print("In final loop n_estimator::",i,"depth::",final_depth,"prev score::",old_score,":current score::",score)
            if score <= old_score:
                if i-5<0:
                    print("******************GradientBoostingRegressor Model reached saturation*************")
                    break
                model=GradientBoostingRegressor(n_estimators=i-5,learning_rate=0.1,max_depth=final_depth,random_state=42)   
                metric,score = regression_model_test(model, X_train, y_train)
                break
            old_score=score

                 
    metrics.append(metric)

    # Feature importance

    imp=permutation_importance(model, X_train, y_train, n_repeats=2, random_state=42)
    if (name == "KNearestNeighbour" or name == "SupportVectorRegression"):
        if name == "KNearestNeighbour":
            imp=permutation_importance(model, X_train, y_train, n_repeats=5, random_state=42)
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": imp.importances_mean,
        }).sort_values("Importance", ascending=False)
        feature_importance[name] = importance
    else:
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        feature_importance[name] = importance
    
    fig = px.line(
        x=list(reg_model_metric.keys()), 
        y=list(reg_model_metric.values()), 
        labels={'x': 'Iteration', 'y': 'Score'},
        title=f'Regression Model - {name} : Behaviour',
        markers=True
    )
    fig.show()

print(terminator)
print("Model Metrics:")
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

for model_name, imp in feature_importance.items():
    print(terminator)
    print(features)
    print(f"Feature Importance for model: {model_name}")
    print(imp)

####### Classification #######
print(terminator)
print("-----------------Classification-----------------")
print(terminator)
#features = ["Protein", "Sugar","Fiber", "Cholesterol", "Serving_Size","Encoded_Preparation_Method"] -- Not a best features to learn
features = ["Protein", "Sugar","Fiber", "Cholesterol", "Serving_Size","Encoded_Food_Name","Encoded_Preparation_Method"]
classification_target = "Is_Gluten_Free" # Is_Vegan accuracy in prediction is not as accurate as Is_Gluten_Free
X = encoded_df[features]
y = encoded_df[classification_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

classification_models = [
    "LogisticRegression",
    "KNeighborsClassifier",
    "Random Forest",
    "SupportVectorClassification",
    "XGBClassifier",
    "GradientBoosting"
]
classification_metrics = []

for name in classification_models:
    print(terminator)
    print(name)
    print(terminator)

    accuracy=0
    model_metric={}
    if name == "LogisticRegression":
        for i in range(10,100,10):
            model = LogisticRegression(solver="lbfgs",max_iter=i)
            acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
            model_metric[i]=acc
            print("prev accuracy::",accuracy,":current acc::",acc)
            if accuracy >= acc:
                model = LogisticRegression(solver="lbfgs",max_iter=i-10)
                acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
                break
            accuracy=acc
    elif name == "SupportVectorClassification":
        model = SVC(kernel='rbf', C=1.0, gamma='scale')
        acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
        model_metric[i]=acc
    
    elif name == "KNeighborsClassifier":
        for i in range(1,10):
            model = KNeighborsClassifier(n_neighbors=i, weights='distance')
            acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
            model_metric[i]=acc
  
    else:
        for i in range(1,10):
            if name == "Random Forest":
                model=RandomForestClassifier(max_depth=i, random_state=42)
            elif name == "XGBClassifier":
                model = xgb.XGBClassifier(n_estimators=i, learning_rate=0.1)
            elif name == "GradientBoosting":
                model=GradientBoostingClassifier(n_estimators=i,learning_rate=0.1,max_depth=3,random_state=42)
            acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
            model_metric[i]=acc
            print("prev accuracy::",accuracy,":current acc::",acc)
            if accuracy >= acc:
                if name == "Random Forest":
                    model=RandomForestClassifier(max_depth=i-1, random_state=42)
                elif name == "XGBClassifier":
                    model = xgb.XGBClassifier(n_estimators=i-1, learning_rate=0.1)
                elif name == "GradientBoosting":
                    model=GradientBoostingClassifier(n_estimators=i-1,learning_rate=0.1,max_depth=3,random_state=42)
                acc,y_pred_ret,classification_metric = classification_model_test(model, X_train, y_train)
                break
            accuracy=acc


    fig = px.line(
        x=list(model_metric.keys()), 
        y=list(model_metric.values()), 
        labels={'x': 'Iteration', 'y': 'Accuracy'},
        title=f'Classification Model - {name} : Behaviour',
        markers=True
    )
    fig.show()


    classification_metrics.append(classification_metric)

    print("Confusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ret).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("")
    print(f'Features used : {features}')
    if (name == "KNeighborsClassifier"):
        imp=permutation_importance(model, X_train, y_train, n_repeats=5, max_samples=0.5, random_state=42)
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": imp.importances_mean,
        }).sort_values("Importance", ascending=False)
        print("Feature Importance:",importance)
    elif (name == "SupportVectorClassification"):
        imp=permutation_importance(model, X_train, y_train, n_repeats=2, max_samples=0.5, random_state=42)
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": imp.importances_mean,
        }).sort_values("Importance", ascending=False)
        print("Feature Importance:",importance)
    elif name == "LogisticRegression":
        print("Feature Importance:",model.coef_)
    else:
        print("Feature Importance:",model.feature_importances_)

#print(classification_report(y_test, y_pred_ret))
classification_metrics_df = pd.DataFrame(classification_metrics)
print(classification_metrics_df)
