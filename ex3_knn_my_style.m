%% 데이터 불러오기 / 각 종을 숫자로 표현 1: versicolor, 2: virginica / 학습데이터와 평가데이터 나누기
clear; close all; clc; 

load fisheriris;
%%
spcs2num = [];
for k=1:1:length(species)
    if strcmp(species(k), 'versicolor') == 1
        spcs2num(k,1) = 1;
    elseif strcmp(species(k), 'virginica') == 1
        spcs2num(k,1) = 2;
    end
end

tr_id = [71:1:100 121:1:150];
Training_data = meas(tr_id, 1:2); % sepal length, width만 가져와라
Training_label = spcs2num(tr_id,:);

ts_id = [51:1:70 101:1:120];
Test_data = meas(ts_id, 1:2); % sepal length, width만 가져와라
Test_label = spcs2num(ts_id,:);

%% 하나씩 해보기: 매트랩 내장함수를 사용하는 것이 아닌, 직접 k 지정부터 거리 계산까지 해보기
k = 3; % k 지정
id = 1;
tmp_data = Test_data(id,:); % 평가해보고 싶은 데이터 가져오기 

% 학습 데이터와 평가해보고 싶은 데이터 간의 거리를 계산
for i=1:1:length(Training_label)
    u_dis(i,1) = sqrt((Training_data(i, 1) - tmp_data(1,1))^2 + (Training_data(i, 2) - tmp_data(1,2))^2);
end

% 가장 인접한 k개를 찾는다
[s_v, s_i] = sort(u_dis, 'ascend');
k_idx = s_i(1:k)

% 해당 k개의 데이터가 어떤 그룹에 속해있는지 확인한다.
k_group = Training_label(k_idx,1)

% 다수결 투표를 통해 최종 그룹 결정: 그룹이 1 또는 2밖에 없으니까
idx1 = find(k_group == 1);
idx2 = find(k_group == 2);

if length(idx1) > length(idx2)
    final_group = 1;
else
    final_group = 2;
end
% final_group = mode(k_group); % mode: 최빈값을 찾는 함수. (참고: 수학에서 mean은 평균, median은 중앙값, mode는 최빈값. 거기서 그 mode임)
[final_group Test_label(id,1)]

%% 모든 test data에 대해 해보기
k = 3; % k 지정
for id = 1:1:length(Test_label)
    tmp_data = Test_data(id,:); % 평가해보고 싶은 데이터 가져오기 
    
    % 학습 데이터와 평가해보고 싶은 데이터 간의 거리를 계산
    for i=1:1:length(Training_label)
        u_dis(i,1) = sqrt((Training_data(i, 1) - tmp_data(1,1))^2 + (Training_data(i, 2) - tmp_data(1,2))^2);
    end
    
    % 가장 인접한 k개를 찾는다
    [s_v, s_i] = sort(u_dis, 'ascend');
    k_idx = s_i(1:k);
    
    % 해당 k개의 데이터가 어떤 그룹에 속해있는지 확인한다.
    k_group = Training_label(k_idx,1);
    final_group(id,1) = mode(k_group);% mode: 최빈값을 찾는 함수. (참고: 수학에서 mean은 평균, median은 중앙값, mode는 최빈값. 거기서 그 mode임)
end

figure;
subplot(311); bar(Test_label); axis tight;
subplot(312); bar(final_group); axis tight;
subplot(313); bar(Test_label - final_group); axis tight;

%% 내가 코드를 잘 구현했다면, 매트랩 내장함수를 이용한 결과와 똑같을 것. 비교해보자
k = 3; % 인접한 이웃 3개를 보겠다.
mdl = fitcknn(Training_data, Training_label, 'NumNeighbors', k, 'Distance', 'euclidean');
mat_res = predict(mdl, Test_data);

figure;
subplot(311); bar(Test_label); axis tight; % 정답
subplot(312); bar(final_group); axis tight;% 내 결과
subplot(313); bar(mat_res); axis tight; % 매트랩 결과