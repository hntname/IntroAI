%% kNN
%clear: 작업공간의 변수를 모두 지워라
% close all: 그림이 떠 있다면 다 지워라
%clc: 명령창에 있는 것들을 다 지워라
clear all; close all; clc;

%붓꽃 데이터 불러오기
%meas 1열: sepal length
%meas 2열: sepal width
%meas 3열: petal length
%meas 4열: petal width
%species: 꽃 종류
load fisheriris; 

%% 특징그려보기
%어떤 값을 분류에 사용할 것인지 보고 결정
figure;
plot(meas(:,1), meas(:,2), 'k.'); 
xlabel('Sepal length');
ylabel('Sepal width');

%% 각 종을 숫자로 표현 1:setosa, 2:vericolor, 3: verginica
spcs2num = []; % 공간만 제작
for k=1:1:length(species) %k는 1부터 length까지 돌아가면서
    if strcmp(species(k),'setosa') ==1 % setosa가 1과 같아지면
        spcs2num(k,1)=1; % setosa 자리에 1 채우기
    elseif strcmp(species(k),'versicolor') == 1
        spcs2num(k,1)=2; % versicolor 자리에 2 채우기
    elseif strcmp(species(k),'virginica') == 1
        spcs2num(k,1)=3; % 3 채우기
    end
end
%% 종별로 다른 색으로 그려보기
idx1 = find(spcs2num==1); % setosa 인덱스만 찾기
idx2 = find(spcs2num==2); % versicolor 인덱스만 찾기
idx3 = find(spcs2num==3); % virginica 인덱스만 찾기

figure;
plot(meas(idx1,1), meas(idx1,2),'r.'); hold on; % setosa는 빨간점
plot(meas(idx2,1), meas(idx2,2),'go'); hold on; % versicolor는 녹색 동그라미
plot(meas(idx3,1), meas(idx3,2),'bx'); hold on; % virginica는 파랑 X
xlabel('Sepal length');
ylabl('Sepal width');

figure;
plot(meas(idx1,3), meas(idx1,4),'r.'); hold on; % setosa는 빨강
plot(meas(idx2,3), meas(idx2,4),'go'); hold on; % versicolor는 녹색
plot(meas(idx3,3), meas(idx3,4),'bx'); hold on; % virginica는 파랑
xlabel('petal length');
yleabl('Sepal width');

%% 학습데이터와 평가데이터 나누기
% 두개의 그룹만 먼저 나누기 versicolor vs verginica
% 오늘은 편의상 아래와 같이 나누겠음
% 1~50: setosa, 51~100: versicolor, 101~150: virginica
% 학습데이터, 71~100: versicolor, 101~150: virginica 총 60개
% 평가데이터, 51~70: versicolor, 101~120: virginica 총 40개

tr_id = [71:1:100 121:1:150]; % training id에서 training data를 가져옴
Training_data = meas(tr_id,:);
Training_label = spcs2num(tr_id,:);

ts_id = [51:1:70 101:1:120];
Test_data = meas(ts_id,:);
Test_label = species(ts_id,:);

