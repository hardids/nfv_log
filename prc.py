import pandas as pd
import re

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


# ["rect_type", "cause", "grade", "Y"]
class Makedata:

    def __init__(self, path):
        df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
        df.recv_c_dttm = pd.to_datetime(df.recv_c_dttm)
        df.create_dttm = pd.to_datetime(df.create_dttm)
        df.resolve_dttm = pd.to_datetime(df.resolve_dttm)
        self.data = df
        self.msg = df.rect_type.values
        self.cause = df.cause.values
        self.grade = df.grade.values


    def load_data(self, path):
        df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
        df.recv_c_dttm = pd.to_datetime(df.recv_c_dttm)
        df.create_dttm = pd.to_datetime(df.create_dttm)
        df.resolve_dttm = pd.to_datetime(df.resolve_dttm)
        self.data = df
        self.msg = df.rect_type.values
        self.cause = df.cause.values
        self.grade = df.grade.values

    def labeling(self):    
        df_ipmi = self.data[self.data.step4.str.contains('ipmi')].copy()
        df_ipmi.rect_type = df_ipmi.rect_type.str.lower()
        df_ipmi["Y"] = 2
        df_ipmi.loc[df_ipmi.grade == 'warning', "Y"] = 0
        df_ipmi.loc[df_ipmi.rect_type.str.contains('completed'), "Y"] = 0
        df_ipmi.loc[df_ipmi.rect_type.str.contains('changed to ok'), "Y"] = 0
        df_ipmi.loc[df_ipmi.rect_type.str.contains('account'), "Y"] = 0
        df_ipmi.loc[df_ipmi.rect_type.str.contains('link fail'), "Y"] = 1
        df_ipmi.loc[df_ipmi.rect_type.str.contains('connection fail'), "Y"] = 1
        df_ipmi.loc[df_ipmi.rect_type.str.contains('fan'), "Y"] = 1
        df_ipmi.loc[df_ipmi.rect_type.str.contains('led'), "Y"] = 1
        df_ipmi.loc[df_ipmi.rect_type.str.contains('chassis'), "Y"] = 1
        df_ipmi.loc[df_ipmi.rect_type.str.contains('correctable memory error'), "Y"] = 2
        self.label_data = df_ipmi
        self.label = df_ipmi.Y.values

    def makeTrainset_text(self):
        
        # X_features = pd.get_dummies(self.label_data[["cause", "grade", "Y"]], columns=["cause", "grade"])
        # drop_df = pd.concat([self.label_data[["rect_type"]], X_features], axis=1).drop_duplicates().copy()
        y = self.label_data.copy().Y.values
        X = self.label_data.copy()[["rect_type"]].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 180)
        # return X_train, X_test, y_train, y_test
        return X,y
    
    def makeTrainset_feature(self):
        
        X_features = pd.get_dummies(self.label_data[["cause", "grade", "Y"]], columns=["cause", "grade"])
        drop_df = pd.concat([self.label_data[["rect_type"]], X_features], axis=1).drop_duplicates().copy()
        y = drop_df.Y.values
        X = drop_df.drop("Y", axis=1).values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 180)
        # return X_train, X_test, y_train, y_test
        return X,y


def preprocessing(log, remove_stopwords = True):

    rm_words = ["slot", "port", "id", "pw"]
    # ????????? ?????? ??????????????? ??????(" ")?????? ?????????
    log = re.sub(r'[^\w]',r' ', log) # ???????????? ??????
    log = re.sub(r'\W*\b\w{1}\b', r'', log) # 1?????? ?????? ??????
    log = log.replace('  ', ' ')
    log = log.strip()

    # ???????????? ???????????? ????????? ?????? ????????? ???????????? ????????? ????????????

    words = log.lower().split()
    words = [w for w in words if not w in rm_words]
    if remove_stopwords:

        # ????????? ??????
        # ?????? ????????? ?????? ????????????
        stops = set(stopwords.words("english"))
        # ???????????? ?????? ????????? ????????? ????????? ????????? ??????
        words = [w for w in words if not w in stops]
        # ?????? ???????????? ????????? ?????? ??????
        clean_review = ' '.join(words)

    else: # ????????? ?????? ????????? ??? ????????????
        clean_review = ' '.join(words)
    return clean_review


if __name__ == "__main__":
    print("hi")