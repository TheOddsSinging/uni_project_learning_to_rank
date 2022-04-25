#!/usr/bin/env python
# coding: utf-8

# In[52]:


import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# In[53]:


tf.compat.v1.disable_eager_execution()


# In[54]:


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """initialize the iterator"""
    
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None
        
    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created"""
        del coord
        self.iterator_initializer_fn(session)


# In[55]:


def example_feature_columns():
    """return the example feature columns"""
    return {
        name:
        tf.feature_column.numeric_column(name, default_value=0.0, shape=(1,))
        for name in feature_names
    }


# In[56]:


def dataframe_transformer_train(df):
    
    """
    transform dataframe into feature_map dict and labels list
    the shape of each feature is [query_num, list_size, 1]
    the shape of labels list is [query_num, list_size]
    feature_map and labels will be convert to tensorflow dataset later
     
    """
    queryIDs = df['#QueryID'].unique() # get the list of unique Query IDs
    df_grouped = df.groupby('#QueryID') # group the dataset by query id 
    
    feature_map = {name:[] for name in feature_names}
    label_list = []
        
    for qid in queryIDs:
            
        df_tmp = df_grouped.get_group(qid)
            
        feature_map_tmp = {name:np.array(df_tmp[name]) for name in feature_names}
        feature_map_tmp = {
            name:
            np.pad(value, (0,list_size-len(value)), 'constant', constant_values=0) 
            for name, value in feature_map_tmp.items()
        }
        feature_map_tmp = {
            name:
            np.reshape(value, (list_size,1))
            for name, value in feature_map_tmp.items()
        }
        for key in feature_map:
            feature_map[key].append(feature_map_tmp[key])
            
        label_list_tmp = np.array(df_tmp['Label'])
        label_list_tmp = np.pad(label_list_tmp, (0, list_size-len(label_list_tmp)), 'constant', constant_values=-1.)
        label_list.append(label_list_tmp)
            
    feature_map = {name:np.array(value).astype('float32') for name, value in feature_map.items()}  # convert the values of the dict to np array
        
    return feature_map, np.array(label_list).astype('float32')


# In[57]:


def dataframe_transformer_pred(df):
    
    queryIDs = df['#QueryID'].unique()
    df_grouped = df.groupby('#QueryID')   
    qid_to_docs = {qid:df_grouped.get_group(qid)['Docid'].values for qid in queryIDs}
   
    feature_map = {name:[] for name in feature_names}
    
    for qid in queryIDs:
        df_tmp = df_grouped.get_group(qid)
        feature_map_tmp = {name:np.array(df_tmp[name]) for name in feature_names}
        feature_map_tmp = {
            name:
            np.pad(value, (0,list_size-len(value)), 'constant', constant_values=0) 
            for name, value in feature_map_tmp.items()
        }
        feature_map_tmp = {
            name:
            np.reshape(value, (list_size,1))
            for name, value in feature_map_tmp.items()
        }
        for key in feature_map:
            feature_map[key].append(feature_map_tmp[key])
            
    feature_map = {name:np.array(value).astype('float32') for name, value in feature_map.items()}
    return feature_map, qid_to_docs


# In[58]:


def get_train_inputs(features, labels, batch_size):
    """set up training input function that meets tensorflow's requirements"""
    iterator_initializer_hook = IteratorInitializerHook()
    
    def _train_input_fn():
        """define training input function"""
    
        # create feature placeholder
        feature_placeholder = {
            key:tf.compat.v1.placeholder(value.dtype, value.shape)
            for key,value in features.items()
        }
    
        # create label placeholder
        label_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
    
        # create dataset that meet tensorflow's requirement
        dataset = tf.data.Dataset.from_tensor_slices((feature_placeholder, label_placeholder))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        
        feed_dict = {label_placeholder:labels}
        feed_dict.update(
            {feature_placeholder[key]:features[key] for key in feature_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()
    
    return _train_input_fn, iterator_initializer_hook


# In[59]:


def get_vali_inputs(features, labels):
    """set up validation inputs in a single batch"""
    iterator_initializer_hook = IteratorInitializerHook()
    
    def _vali_input_fn():
        """define validation input function"""
        feature_placeholder = {
            key:tf.compat.v1.placeholder(value.dtype, value.shape)
            for key,value in features.items()
        }
        label_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensors((feature_placeholder, label_placeholder))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        feed_dict = {label_placeholder:labels}
        feed_dict.update(
            {feature_placeholder[key]:features[key] for key in feature_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()
    
    return _vali_input_fn, iterator_initializer_hook


# In[60]:


def get_predict_inputs(features):
    
    iterator_initializer_hook = IteratorInitializerHook()
    
    def _predict_input_fn():
        
        feature_placeholder = {
            key:tf.compat.v1.placeholder(value.dtype, value.shape)
            for key,value in features.items()
        }
        dataset = tf.data.Dataset.from_tensors(feature_placeholder)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        feed_dict = {}
        feed_dict.update(
            {feature_placeholder[key]:features[key] for key in feature_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()
    
    return _predict_input_fn, iterator_initializer_hook


# In[61]:


def make_serving_input_fn():
    """return srving input fn to receive tf.Example"""
    feature_spec = tf.feature_column.make_parse_example_spec(example_feature_columns().values())
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)


# In[62]:


def make_transform_fn():
    """returns a transform fuction that converts features to dense tensors"""
    
    def _transform_fn(features, mode):
        """define transform_fn"""
        context_features, example_features = (
            tfr.feature.encode_listwise_features(
                features=features, 
                context_feature_columns=None, 
                example_feature_columns=example_feature_columns(), 
                mode=mode, 
                scope='transform_layer'))
            
        return context_features, example_features
    
    return _transform_fn


# In[63]:


def make_score_fn(hparams):
    """return a score function"""
    
    def _score_fn(context_features, group_features, mode, params, config):
        """define the network to score documents"""
        with tf.compat.v1.name_scope('input_layer'):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.compat.v1.summary.scalar('input_sparsity', tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar('input_max', tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar('input_min', tf.reduce_min(input_tensor=input_layer))
            
        is_training = (mode==tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.compat.v1.layers.batch_normalization(input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in hparams['hidden_layer_dims']):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar('fully_connected_{}_sparsity'.format(i), tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.compat.v1.layers.dropout(cur_layer, rate=hparams['dropout_rate'], training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=group_size)
        return logits
    
    return _score_fn


# In[64]:


def get_eval_metric_fn():
    """return a dict from name to metric functions"""
    metric_fns = {}
#     metric_fns.update({
#         'metric/%s' % name:tfr.metrics.make_ranking_metric_fn(name) for name in [
#             tfr.metrics.RankingMetricKey.ARP, 
#             tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY
#         ]
#     })
    metric_fns.update({
        'metric/ndcg%d' % topn:tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [10, 20, 100]
    })
    return metric_fns


# In[65]:


def train_and_eval(features_train, labels_train, features_vali, labels_vali, hparams):
    """train and evaluate"""
    
    train_input_fn, train_hook = get_train_inputs(features_train, labels_train, batch_size)
    vali_input_fn, vali_hook = get_vali_inputs(features_vali, labels_vali)
    
    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=hparams['learning_rate'])
    
    def _train_op_fn(loss):
        """define train op used in ranking head"""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([minimize_op, update_ops])
        return train_op
    
    ranking_head = tfr.head.create_ranking_head(
        loss_fn=tfr.losses.make_loss_fn(loss), 
        eval_metric_fns=get_eval_metric_fn(), 
        train_op_fn=_train_op_fn)
       
    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(hparams), 
            group_size=group_size, 
            transform_fn=make_transform_fn(), 
            ranking_head=ranking_head), 
#         model_dir=model_dir, 
        config=tf.estimator.RunConfig(save_checkpoints_steps=1000))
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        hooks=[train_hook], 
        max_steps=num_train_steps)
    
    vali_spec = tf.estimator.EvalSpec(
        input_fn=vali_input_fn, 
        hooks=[vali_hook], 
        steps=1, 
        exporters=tf.estimator.LatestExporter(
            'latest_exporter', 
            serving_input_receiver_fn=make_serving_input_fn()),
        start_delay_secs=0, 
        throttle_secs=30)
    

    return (estimator, train_spec, vali_spec)


# In[66]:


def tuning_and_cross_validation(file_path, tuning_opts, cv=5):
    
    df = pd.read_csv(file_path, sep='\t')

    # hyperparamers tuning
    tuning_results = {}
    for hidden_layer_dims in tuning_opts['hidden_layer_dims']:
        for learning_rate in tuning_opts['learning_rate']:
            for dropout_rate in tuning_opts['dropout_rate']:
                hparams = {
                    'hidden_layer_dims':hidden_layer_dims,
                    'learning_rate':learning_rate,
                    'dropout_rate':dropout_rate
                }
                
                # kfold cross validation
                kf = KFold(n_splits=cv, random_state=23, shuffle=True)
                ndcg100 = []

                for train_idx, vali_idx in kf.split(df):

                    train = df.loc[train_idx]
                    vali = df.loc[vali_idx]

                    features_train, labels_train = dataframe_transformer_train(train)
                    features_vali, labels_vali = dataframe_transformer_train(vali)

                    estimator, train_spec, vali_spec = train_and_eval(features_train, 
                                                                      labels_train, 
                                                                      features_vali, 
                                                                      labels_vali, 
                                                                      hparams)
                    ndcg, path = tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)
                    ndcg100.append(ndcg['metric/ndcg100'])

                tuning_results.update({
                    str(hparams['hidden_layer_dims']) + ':' +
                    str(hparams['learning_rate']) + ":" +
                    str(hparams['dropout_rate']):
                    np.array(ndcg100).mean()})

    return tuning_results, estimator

    
    
# estimator, train_spec, vali_spec = train_and_eval()
#train and validate
# accuracy = tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)


# In[67]:


def best_hparams(tuning_results):
    for key, value in tuning_results.items():
        if value == max(tuning_results.values()):
            print(key)


# In[68]:


if __name__=='__main__':
    ## define constants for later use
    feature_names = [
        'BodyTerms', 'AnchorTerms', 'TitleTerms', 'URLTerms', 'TermsWholeDocument', 'IDFBody', 'IDFAnchor', 'IDFTitle', 
        'IDFURL', 'IDFWholeDocument', 'TFIDFBody', 'TFIDFAnchor', 'TFIDFTitle', 'TFIDFURL', 'TFIDFWholeDocument', 'LengthBody', 
        'LengthAnchor', 'LengthTitle', 'LengthURL', 'LengthWholeDocument', 'BM25Body', 'BM25Anchor', 'BM25Title', 'BM25URL', 
        'BM25WholeDocument', 'LMIRABSBody', 'LMIRABSAnchor', 'LMIRABSTitle', 'LMIRABSURL', 'LMIRABSWholeDocument', 
        'LMIRDIRBody', 'LMIRDIRAnchor', 'LMIRDIRTitle', 'LMIRDIRURL', 'LMIRDIRWholeDocument', 'LMIRIMBody', 'LMIRIMAnchor', 
        'LMIRIMTitle', 'LMIRIMURL', 'LMIRIMWholeDocument', 'PageRank', 'InlinkNum', 'OutlinkNum', 'NumSlashURL', 'LenURL', 
        'NumChildPages'
    ]
    label_name = 'Label'
    list_size = 147 # maximum number of documents correspond to a single query
    batch_size = 32
    num_train_steps = 3000
    group_size = 1
    loss = 'approx_ndcg_loss' 
    train_path = 'train.tsv'
    test_path = 'test.tsv'
    tuning_opts = {
        'hidden_layer_dims':[['64', '32', '16'], ['256', '128']], 
        'learning_rate':[0.01, 0.05],
        'dropout_rate':[0.4, 0.8] 
    }
    
    
    tuning_results, estimator = tuning_and_cross_validation(train_path, tuning_opts, cv=5)
    best_hparams(tuning_results)
    best = {'hidden_layer_dims':['256', '128'], 'learning_rate':0.05, 'dropout_rate':0.8}
    
    df_train = pd.read_csv(train_path, sep='\t')
    train = df_train.head(60000)
    vali = df_train.tail(14216)
    features_train, labels_train = dataframe_transformer_train(train)
    features_vali, labels_vali = dataframe_transformer_train(vali)

    estimator, train_spec, vali_spec = train_and_eval(features_train, labels_train, features_vali, labels_vali, best)
    tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)
    
    df_test = pd.read_csv(test_path, sep='\t')
    features_pred, qid_to_docs = dataframe_transformer_pred(df_test)
    predict_input_fn, pred_hook = get_predict_inputs(features_pred)
    pred = estimator.predict(input_fn=predict_input_fn, hooks=[pred_hook])
    pred_list = list(pred)
    qid_docid_score_list = [['\t'.join([qid, docid, str(score)]) 
                             for docid, score in zip(qid_to_docs[qid], pred_list[idx])]
                            for idx, qid in enumerate(qid_to_docs.keys())]
    qid_docid_score_list = ['\n'.join(sub_list) for sub_list in qid_docid_score_list]
    qid_docid_score_list = '\n'.join(qid_docid_score_list)
    with open('A2.tsv', 'w') as f:
        f.write(qid_docid_score_list)

