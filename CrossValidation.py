from sklearn.model_selection import KFold
import sklearn.metrics

#Perform Kfold Cross validation. Return array of accuracies for each fold
def KCrossValidation(model, inputs, labels, n_folds=5, seed=42, type='ml', pretrained=False):
    fold_num = 1
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for train, test in kfold.split(inputs,labels):
        print("Fold Number: ", fold_num)
        if not pretrained:
            model.fit(inputs[train], labels[train])

        if(type=='ml'):
            predictions = model.predict(inputs[test])
            accuracy = sklearn.metrics.accuracy_score(labels[test], predictions)*100
        else:
            scores = model.evaluate(inputs[test], labels[test])
            accuracy = scores[1]*100

        print('Accuracy: %.2f%%' % accuracy)
        results.append(accuracy)
        fold_num += 1
    return results
    
