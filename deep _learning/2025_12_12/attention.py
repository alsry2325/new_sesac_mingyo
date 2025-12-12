import numpy as np


#Softmax 함수 + Softmax의 역전파(backward)
def softmax(x): #여러 값(로짓)을 확률처럼 0~1 사이로 변환하는 함수.
    if x.ndim == 2:  #2차원 입력 (배치 데이터)
        x = x - x.max(axis=1, keepdims=True)  #최댓값을 빼서 안정적인 계산 (수치 폭발 방지)
        x = np.exp(x)   #exponent 적용 → exp(x)
        x /= x.sum(axis=1, keepdims=True)  #각 행(row)별 합으로 나눔 → 확률값
    elif x.ndim == 1:  #1차원 입력 (단일 데이터)
        x = x - np.max(x)   #하나의 벡터에 대해 softmax 계산
        x = np.exp(x) / np.sum(np.exp(x))

    return x

class Softmax:
    #Softmax는 학습 파라미터가 없는 함수라 빈 리스트.
    def init(self):
        self.params, self.grads = [], []
        self.out = None
#순전파
    def forward(self, x):  #입력 x → softmax 계산 결과 out 저장. ,로짓을 확률로 바꾸는 함수다.
        self.out = softmax(x)
        return self.out
#역전파  끝에서 들어온 기울기를 가지고 각 입력에 대한 변화량을 계산한다.”
    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


N, T, H = 3, 5, 4 
#N : 샘플(문장) 개수
#T : 각 문장의 단어/토큰 개수 (타임스텝 개수),시간축 
#H : 단어의 의미를 담은 숫자 벡터의 크기 
 
#(N, T, H) 모양의 랜덤 벡터 시퀀스를 만든다.,정규분포(평균 0, 표준편차 1)의 랜덤 숫자로 채운다
hs = np.random.randn(N, T, H) 
# print(hs)

##(N,H) 모양의 랜덤 벡터 시퀀스를 만든다.,정규분포(평균 0, 표준편차 1)의 랜덤 숫자로 채운다 # 이차원 배열
h = np.random.randn(N, H)

#문장당 단어 1개짜리 시퀀스처럼 보이도록" 삼차원으로 바꿈
h_reshape = h.reshape(N, 1, H)
# print(h)

#트랜스포머 Attention 계산
#“전체 시퀀스에 대해 한 단어(또는 한 은닉 상태)를 비교하기 위해 T번 복사한다” axis=1 : ~에따라 계산을 하라는 의미 여기서는 시간축 
hr = h_reshape.repeat(T, axis=1)

#전체 단어의 은닉 상태 * 한 단어의 은닉 상태를 T번 복사한 것 
#즉 한 단어와 모든 단어의 벡터를 원소별로 곱해서 유사도(가중치)계산의 기초 정보를  만드는 단계
t1 = hs * hr

# a + b + c + d  → 단어 1개에 대해 숫자 1개로 요약, 벡터의 각 원소를 다 더해서, 하나의 스칼라(숫자)로 요약한다.
score = np.sum(t1, axis=2) 

#score(유사도 점수)를 Softmax로 확률로 변환한 값을 a에 저장
a = Softmax().forward(score)
print(a)

#각 문장별 Softmax 확률이 1이 되는지 확인하는 코드이다.
a_sum = a.sum(axis=1)
print(a_sum)


#각 시간마다 1개의 숫자 형태로 만들고,hidden size(H) 만큼 복제해서 (N,T,H)로 맞춘다.
#attention weight를 hidden state와 곱할 수 있는 형태
ar = a.reshape(N, T, 1).repeat(H, axis=2)
print(ar)

#해당 단어의 중요도(가중치)
print(ar.shape)
#전체 문장들
print(hs.shape)

#각 히든 스테이트(hs)에 attention weight(ar)를 곱해서 중요도를 반영하는 과정” Attention이 적용된 히든 표현
t = hs * ar
print(t)

#각 단어에 가중치를 곱해 만든 t 를 시간축(T)으로 모두 더해서 ,문장 전체를 하나의 벡터로 요약한 것 → 바로 컨텍스트 벡터(context vector)
c = np.sum(t, axis=1)

#결론은
'''
중요한 단어는 크게 반영하고 , 덜 중요한 단어는  작게 반영하는 문장 요약 벡터 를 만드는 과정
'''
