from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator


#Create a Logistic Regression classifier.
def lr_train(train):
    lr = LogisticRegression(labelCol = "isFraud", featuresCol = "features")
    # Learn from the training data.
    lrModel = lr.fit(train)
    return lrModel


def lr_eval_test(lrModel, test):
    prediction_LR = lrModel.transform(test)
    prediction_LR.groupBy("isFraud", "prediction").count().show()
    tp = prediction_LR[(prediction_LR.isFraud == 1) & (prediction_LR.prediction == 1)].count()
    tn = prediction_LR[(prediction_LR.isFraud == 0) & (prediction_LR.prediction == 0)].count()
    fp = prediction_LR[(prediction_LR.isFraud == 0) & (prediction_LR.prediction == 1)].count()
    fn = prediction_LR[(prediction_LR.isFraud == 1) & (prediction_LR.prediction == 0)].count()
    recall_LR = tp/(tp+fn)
    precision_LR = tp/(tp+fp)
    f1_score_LR = 2*(recall_LR*precision_LR)/(recall_LR+precision_LR)
    print("Recall : ",recall_LR)
    print("Precision : ", precision_LR)
    print("F1 Score : ", f1_score_LR)
    # Area under ROC curve
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
    areaUnderROC_LR = evaluator.evaluate(prediction_LR, {evaluator.metricName: "areaUnderROC"})
    print("Area under ROC = %s" % areaUnderROC_LR)
    # Area under precision-recall curve
    areaUnderPR_LR = evaluator.evaluate(prediction_LR, {evaluator.metricName: "areaUnderPR"})
    print("Area under PR = %s" % areaUnderPR_LR)
    results = {}
    results['Logestic Regression'] = [recall_LR, precision_LR, f1_score_LR, areaUnderROC_LR, areaUnderPR_LR]
    return results


def Dt_train(train):
    # Using the DecisionTree classifier model
    dt = DecisionTreeClassifier(labelCol = "isFraud", featuresCol = "features", seed = 54321, maxDepth = 5)
    dt_model = dt.fit(train)
    return dt_model

def Dt_eval_test(dt_model,test):
    prediction_DT = dt_model.transform(test)
    # Select example rows to display.
    prediction_DT.select('features', 'rawPrediction', 'probability', 'prediction').show(5)
    prediction_DT.groupBy("isFraud", "prediction").count().show()
    tp = prediction_DT[(prediction_DT.isFraud == 1) & (prediction_DT.prediction == 1)].count()
    tn = prediction_DT[(prediction_DT.isFraud == 0) & (prediction_DT.prediction == 0)].count()
    fp = prediction_DT[(prediction_DT.isFraud == 0) & (prediction_DT.prediction == 1)].count()
    fn = prediction_DT[(prediction_DT.isFraud == 1) & (prediction_DT.prediction == 0)].count()
    recall_DT = tp/(tp+fn)
    precision_DT = tp/(tp+fp)
    f1_score_DT = 2*(recall_DT*precision_DT)/(recall_DT+precision_DT)
    print("Recall : ",recall_DT)
    print("Precision : ", precision_DT)
    print("F1 Score : ", f1_score_DT)
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
    # Area under ROC curve
    areaUnderROC_DT = evaluator.evaluate(prediction_DT, {evaluator.metricName: "areaUnderROC"})
    print("Area under ROC = %s" % areaUnderROC_DT)
    # Area under precision-recall curve
    areaUnderPR_DT = evaluator.evaluate(prediction_DT, {evaluator.metricName: "areaUnderPR"})
    print("Area under PR = %s" % areaUnderPR_DT)
    results = {}
    results['Decision Tree Classifier'] = [recall_DT, precision_DT, f1_score_DT, areaUnderROC_DT, areaUnderPR_DT]
    return results


def rf_train(train):
    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=10)
    rf_model = rf.fit(train)
    return rf_model

def rf_eval_test(rf_model, test):
    # Make predictions.
    prediction_RF = rf_model.transform(test)
    prediction_RF.groupBy("isFraud", "prediction").count().show()

    tp = prediction_RF[(prediction_RF.isFraud == 1) & (prediction_RF.prediction == 1)].count()
    tn = prediction_RF[(prediction_RF.isFraud == 0) & (prediction_RF.prediction == 0)].count()
    fp = prediction_RF[(prediction_RF.isFraud == 0) & (prediction_RF.prediction == 1)].count()
    fn = prediction_RF[(prediction_RF.isFraud == 1) & (prediction_RF.prediction == 0)].count()
    recall_RF = tp/(tp+fn)
    precision_RF = tp/(tp+fp)
    f1_score_RF = 2*(recall_RF*precision_RF)/(recall_RF+precision_RF)
    print("Recall : ",recall_RF)
    print("Precision : ", precision_RF)
    print("F1 Score : ", f1_score_RF)
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
    # Area under ROC curve
    areaUnderROC_RF = evaluator.evaluate(prediction_RF, {evaluator.metricName: "areaUnderROC"})
    print("Area under ROC = %s" % areaUnderROC_RF)
    # Area under precision-recall curve
    areaUnderPR_RF = evaluator.evaluate(prediction_RF, {evaluator.metricName: "areaUnderPR"})
    print("Area under PR = %s" % areaUnderPR_RF)
    results = {}
    results['Random Forest Classifier'] = [recall_RF, precision_RF, f1_score_RF, areaUnderROC_RF, areaUnderPR_RF]
    return results


def GBT_train(train):
    # Train a GBT model.
    gbt = GBTClassifier(labelCol="isFraud", featuresCol="features", maxIter=10)
    # Train model. 
    gbt_model = gbt.fit(train)
    return gbt_model


def GBT_eval_test(gbt_model,test):
    # Make predictions.
    prediction_GBT = gbt_model.transform(test)
    prediction_GBT.groupBy("isFraud", "prediction").count().show()
    tp = prediction_GBT[(prediction_GBT.isFraud == 1) & (prediction_GBT.prediction == 1)].count()
    tn = prediction_GBT[(prediction_GBT.isFraud == 0) & (prediction_GBT.prediction == 0)].count()
    fp = prediction_GBT[(prediction_GBT.isFraud == 0) & (prediction_GBT.prediction == 1)].count()
    fn = prediction_GBT[(prediction_GBT.isFraud == 1) & (prediction_GBT.prediction == 0)].count()
    recall_GBT = tp/(tp+fn)
    precision_GBT = tp/(tp+fp)
    f1_score_GBT = 2*(recall_GBT*precision_GBT)/(recall_GBT+precision_GBT)
    print("Recall : ",recall_GBT)
    print("Precision : ", precision_GBT)
    print("F1 Score : ", f1_score_GBT)
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
    # Area under ROC curve
    areaUnderROC_GBT = evaluator.evaluate(prediction_GBT, {evaluator.metricName: "areaUnderROC"})
    print("Area under ROC = %s" % areaUnderROC_GBT)
    # Area under precision-recall curve
    areaUnderPR_GBT = evaluator.evaluate(prediction_GBT, {evaluator.metricName: "areaUnderPR"})
    print("Area under PR = %s" % areaUnderPR_GBT)
    results = {}
    results['Gradient-Boosted Tree Classifier'] = [recall_GBT, precision_GBT, f1_score_GBT, areaUnderROC_GBT, areaUnderPR_GBT]
    return results


def NB_train(train):
    # create the trainer and set its parameters
    nb = NaiveBayes(labelCol="isFraud", featuresCol="features", smoothing=1.0, modelType="multinomial")
    # train the model
    nb_model = nb.fit(train)
    return nb_model


def NB_eval_test(nb_model,test):
    # Make predictions.
    prediction_NB = nb_model.transform(test)
    prediction_NB.groupBy("isFraud", "prediction").count().show()
    tp = prediction_NB[(prediction_NB.isFraud == 1) & (prediction_NB.prediction == 1)].count()
    tn = prediction_NB[(prediction_NB.isFraud == 0) & (prediction_NB.prediction == 0)].count()
    fp = prediction_NB[(prediction_NB.isFraud == 0) & (prediction_NB.prediction == 1)].count()
    fn = prediction_NB[(prediction_NB.isFraud == 1) & (prediction_NB.prediction == 0)].count()
    recall_NB = tp/(tp+fn)
    precision_NB = tp/(tp+fp)
    f1_score_NB = 2*(recall_NB*precision_NB)/(recall_NB+precision_NB)
    print("Recall : ",recall_NB)
    print("Precision : ", precision_NB)
    print("F1 Score : ", f1_score_NB)
    evaluator = BinaryClassificationEvaluator(labelCol="isFraud")
    # Area under ROC curve
    areaUnderROC_NB = evaluator.evaluate(prediction_NB, {evaluator.metricName: "areaUnderROC"})
    print("Area under ROC = %s" % areaUnderROC_NB)
    # Area under precision-recall curve
    areaUnderPR_NB = evaluator.evaluate(prediction_NB, {evaluator.metricName: "areaUnderPR"})
    print("Area under PR = %s" % areaUnderPR_NB)
    results = {}
    results['Naive Bayes'] = [recall_NB, precision_NB, f1_score_NB, areaUnderROC_NB, areaUnderPR_NB]
    return results
    
    
