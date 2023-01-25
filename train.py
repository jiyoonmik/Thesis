import functions as f
import models as m        
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold

X_seq, X_f, target, grp = f.makeXY()
X_lf = f.make_lf(X_seq)

evalscores={"model1_1" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model2_1" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model2_2" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model2_3" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model3_1" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model3_2" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model3_3" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model4_1" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model4_2" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []},
        "model4_3" : {"val_loss" : [],
                    "val_acc" : [],
                    "val_f1" : [],
                    "timecost" : []}}


skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
fold_var=1

for train_idx, val_idx in skf.split(X_seq, grp):

    X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
    X_f_train, X_f_val = X_f[train_idx], X_f[val_idx]
    Y_train, Y_val = target[train_idx], target[val_idx]

    X_seq_scaled_train, X_seq_scaled_test = f.scale_data(X_seq_train, X_seq_val, 4000)
    X_sv_train_scaled = X_seq_scaled_train[:,:,1:]
    X_sv_test_scaled = X_seq_scaled_test[:,:,1:]


    X_f_scaled_train, X_f_scaled_test = f.scale_data(X_f_train, X_f_val, 152)
    X_f_scaled_train, X_f_scaled_test = f.scale_data(X_f_train, X_f_val, 152)
    X_tf_train_scaled = X_f_scaled_train[:,:,0].reshape(X_f_train.shape[0], 152, 1)
    X_tf_test_scaled = X_f_scaled_test[:,:,0].reshape(X_f_val.shape[0], 152, 1)
    X_idf_train_scaled = X_f_scaled_train[:,:,1].reshape(X_f_train.shape[0], 152, 1)
    X_idf_test_scaled = X_f_scaled_test[:,:,1].reshape(X_f_val.shape[0], 152, 1)
    X_af_train, X_af_val = X_lf[train_idx], X_lf[val_idx]
    X_af_scaled_train, X_af_scaled_test = f.scale_data(X_af_train, X_af_val, 41)

    models={"model1_1" : {"trainX" : X_seq_scaled_train[:,:,1:], 
                        "testX" : X_seq_scaled_test[:,:,1:], 
                        "model" : m.seqModel(seq_n=2, modelnum=1), 
                        "bos_no" : 0},
            "model2_1" : {"trainX" : X_f_scaled_train,
                        "testX" : X_f_scaled_test, 
                        "model" : m.CNNmodel(2,1), 
                        "bos_no" : 2},
            "model2_2" : {"trainX" : X_tf_train_scaled,
                        "testX" : X_tf_test_scaled, 
                        "model" : m.CNNmodel(1,2), 
                        "bos_no" : 1},
            "model2_3" : {"trainX" : X_idf_train_scaled,
                        "testX" : X_idf_test_scaled, 
                        "model" : m.CNNmodel(1,3), 
                        "bos_no" : 1},
            "model3_1" : {"trainX" : [X_sv_train_scaled, X_f_scaled_train],
                        "testX" : [X_sv_test_scaled, X_f_scaled_test], 
                        "model" : m.Ensemble(2,2,1), 
                        "bos_no" : 2},
            "model3_2" : {"trainX" : [X_sv_train_scaled, X_tf_train_scaled], 
                        "testX" : [X_sv_test_scaled, X_tf_test_scaled],
                        "model" : m.Ensemble(2,1,1), 
                        "bos_no" : 1},
            "model3_3" : {"trainX" : [X_sv_train_scaled, X_idf_train_scaled], 
                        "testX" : [X_sv_test_scaled, X_idf_test_scaled], 
                        "model" : m.Ensemble(2,1,2),
                        "bos_no" : 1},
            "model4_1" : {"trainX" : X_seq_scaled_train, 
                        "testX" : X_seq_scaled_test, 
                        "model" : m.seqModel(seq_n=3, modelnum=4),
                        "bos_no" : 0},
            "model4_2" : {"trainX" : X_af_scaled_train, 
                        "testX" : X_af_scaled_test, 
                        "model" : m.labelfeature(), 
                        "bos_no" : 0},
            "model4_3" : {"trainX" : [X_seq_scaled_train, X_af_scaled_train],
                        "testX" : [X_seq_scaled_test, X_af_scaled_test],
                        "model" : m.label_ensemble(seq_n=3), 
                        "bos_no" : 0}}

    for model in models: 
        model, history, train_time = f.run_history(models[model]["model"], models[model]["trainX"], Y_train, fold_var, epoch=1, bat=1)
        f.show_his(history)

        evalscores[model]['timecost'].append(train_time)
        # evaluate the model
        scores = model.evaluate(models[model]["testX"], Y_val)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        evalscores[model]['val_loss'].append(scores[0])
        evalscores[model]['val_acc'].append(scores[1])
        evalscores[model]['val_f1'].append(scores[2])

        tf.keras.backend.clear_session()

        fold_var+=1

for model in evalscores: 
    val_loss, val_acc, val_f1, timecost = f.measures(evalscores[model]['val_loss'], evalscores[model]['val_acc'], evalscores[model]['val_f1'], evalscores[model]['timecost'])