import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

# 개체명 인식 모듈
class NerModel:
    def __init__(self, model_name, preprocess):

        # BIO 태그 클래스별 레이블
        # 개체이름 = 인명(PER),지명(LOC),기관명(ORG),기타(POH)
        # 시간표현 = 날짜(DAT),시간(TIM),기간(DUR)
        # 수량표현 = 통화(MNY),비율(PNT),기타 수량표현(NOH)
        self.index_to_ner = {1: 'O', 2: 'B-PER', 3: 'B-LOC', 4: 'B-ORG', 5: 'B-POH', 6: 'B-DAT', 7: 'B-TIM',
                             8: 'B-DUR', 9: 'B-MNY', 10: 'B-PNT', 11: 'B-NOH', 12: 'I-PER', 13: 'I-LOC', 14: 'I-ORG',
                             15: 'I-POH', 16: 'I-DAT', 17: 'I-TIM', 18: 'I-DUR', 19: 'I-MNY', 20: 'I-PNT', 21: 'I-NOH',
                             22: 'PAD'}

        # 의도 분류 모델 불러오기
        self.model = load_model(model_name)

        # 챗봇 Preprocess 객체
        self.p = preprocess

    # 객체명 클래스 예측
    def predict(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장 내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 패딩 처리
        max_len = 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, padding='post', value=0, maxlen=max_len)

        # 키워드별 개체명 예측
        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)

        tags = [self.index_to_ner[i] for i in predict_class.numpy()[0]]
        return list(zip(keywords, tags))

    def predict_tags(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장 내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 패딩 처리
        max_len = 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, padding='post', value=0, maxlen=max_len)

        predict = self.model.predict(np.array([padded_seqs[0]]))
        predict_class = tf.math.argmax(predict, axis=-1)

        tags = []
        for tag_idx in predict_class.numpy()[0]:
            if tag_idx == 1: continue
            tags.append(self.index_to_ner[tag_idx])
            if len(tags) == 0:
                return None
            return tags
