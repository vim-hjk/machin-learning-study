import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import Binarizer


# UCI HAR Dataset

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)

    return new_feature_name_df

def get_human_dataset():

    # 각 데이터 파일은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를 sep으로 할당.
    feature_name_df = pd.read_csv('../part1/UCI HAR Dataset/features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

    # 중복된 feature명을 수정하는 get_new_feature_name_df()를 이용, 신규 feature명 DataFrame 생성.
    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    # DataFrame에 feature명을 column으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    # 학습 feature dataset과 테스트 feature dataset을 DataFrame으로 loading, column명은 feature_name 적용
    X_train = pd.read_csv('../part1/UCI HAR Dataset/train/X_train.txt', sep='\s+', names=feature_name)
    X_test = pd.read_csv('../part1/UCI HAR Dataset/test/X_test.txt', sep='\s+', names=feature_name)

    # 학습 label과 테스트 label 데이터를 DataFrame으로 loading, column명은 action으로 부여
    y_train = pd.read_csv('../part1/UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv('../part1/UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names=['action'])

    # load된 학습/테스트용 DataFrame을 모두 반환
    return X_train, X_test, y_train, y_test



# Custom get_clf_eval
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    # threshold list 객체 내의 값을 차례로 iterable하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1.reshape(-1, 1))
        custom_predict = binarizer.transform(pred_proba_c1.reshape(-1, 1))
        print('임곗값 : ', custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)