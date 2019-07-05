Pkg.add(“DataFrames”)
Pkg.add("CSV")
Pkg.add("ScikitLearn")
using DataFrames, CSV, ScikitLearn: fit!, predict, @sk_import, fit_transform! 
@sk_import preprocessing: LabelEncoder 
 @sk_import model_selection: cross_val_score  
 @sk_import metrics: accuracy_score 
 @sk_import linear_model: LogisticRegression 
 @sk_import ensemble: RandomForestClassifier 
 @sk_import tree: DecisionTreeClassifier 

##change directory to the folder containing test and train files
;cd /path to the train file
train = CSV.read("........train file..........") // readtable has been deprecated. 
test  = CSV.read("........test file ..........")
describe(train) // gives us the "missing" values info ; showcols() has been deprecated
##scikitlearn uses PyCall.jl and pyhton doesn't support the data type "missing" , so we fill out the missing data
train[ismissing.train[://any column]), ://column] = mode(dropmissing(train[://any column]))    //isna and dropna didn't work for me

##Another problem I faced was, version mismatch of python.
##ScikitLearn.jl installed  and minimal Python 3.7 while I had my default python enivronment set to Pyhton 3 which was recognised by PyCall.
##You can fix this:
   In julia terminal type 
                ENV["PYTHON"] = ""   //This will set the python environment to Julia's i.e. the one installed by ScikitLearn.jl
                restart julia and in the terminal type
                using PyCall
                PyCall.conda
                If it returns true you're good to go.


function classification_model(model, predictors) 
     y = convert(Array, train[:13]) 
     X = convert(Array, train[predictors]) 
     X2 = convert(Array, test[predictors])                  
     
    #Fit the model: 
     fit!(model, X, y) 

     #Make predictions on training set: 
     predictions = predict(model, X) 

     #Print accuracy 
     accuracy = accuracy_score(predictions, y) 
     println("\naccuracy: ",accuracy) 

     #5 fold cross validation 
     cross_score = cross_val_score(model, X, y, cv=5)    
 
     #print cross_val_score 
     println("cross_validation_score: ", mean(cross_score)) 

     #return predictions 
     fit!(model, X, y) 
     global pred = predict(model, X2)  //declaring scope of pred as global so that it can be written into a csv file
     return pred 
 end

model = //any
predictors =[:Gender, :Married, :Dependents, :Education,
 :Self_Employed, :Loan_Amount_Term, :Credit_History, :Property_Area,
 :LoanAmount]  //you may add any nuber of features
classification_model(model, predictors)

sample_submit=CSV.read("sample_submission.csv", copycols = true)

submission = [ test[: , 1] , pred]
CSV.write("submission.csv" , submission )


