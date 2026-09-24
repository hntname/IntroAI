%% 공간이 어떻게 만들어질까?
clear; close all; clc;

load fisheriris;

spcs2num = [];
for k=1:1:length(species)
    if strcmp(species(k), 'setosa') == 1
        spcs2num(k,1) = 1;
    elseif strcmp(species(k), 'versicolor') == 1
        spcs2num(k,1) = 2;
    elseif strcmp(species(k), 'virginica') == 1
        spcs2num(k,1) = 3;
    end
end

idx1 = find(spcs2num == 1);
idx2 = find(spcs2num == 2);
idx3 = find(spcs2num == 3);

figure;
plot(meas(idx1,1), meas(idx1,2), 'r.'); hold on; 
plot(meas(idx2,1), meas(idx2,2), 'g.'); hold on; 
plot(meas(idx3,1), meas(idx3,2), 'b.'); hold on; 
xlabel('Sepal length');
ylabel('Sepal width');


tr_id = [21:1:50 71:1:100 121:1:150];
Training_data = meas(tr_id,1:2); 
Training_label = spcs2num(tr_id,:);

tmp1 = min(meas(:,1)):0.01:max(meas(:,1));
tmp2 = linspace(min(meas(:,2)), max(meas(:,2)), length(tmp1)); % linspnace(a, b, c) a부터 b까지 c개가 되도록 데이터를 생성하는 함수

[out1,out2] = meshgrid(tmp1,tmp2);

Test_data = [];
Test_data = [out1(:) out2(:)];

figure; plot(Test_data(:,1), Test_data(:,2), '.');

%%

k = 50; % kNN의 k를 지정

final_group = [];
for id = 1:1:length(Test_data)
    tmp_data = Test_data(id,:); % 평가해보고 싶은 데이터 가져오기 
    
    % 학습 데이터와 평가해보고 싶은 데이터 간의 거리를 계산
    u_dis = [];
    for i=1:1:length(Training_label)
        u_dis(i,1) = sqrt( sum( (Training_data(i,:) - tmp_data).^2 ) );
    end
    
    % 가장 인접한 k개를 찾는다
    [s_v, s_i] = sort(u_dis, 'ascend');
    k_idx = s_i(1:k);
    
    % 해당 k개의 데이터가 어떤 그룹에 속해있는지 확인한다.
    k_group = Training_label(k_idx,1);
    final_group(id,1) = mode(k_group);% mode: 최빈값을 찾는 함수. (참고: 수학에서 mean은 평균, median은 중앙값, mode는 최빈값. 거기서 그 mode임)
end

idx1 = find(final_group == 1);
idx2 = find(final_group == 2);
idx3 = find(final_group == 3);

figure;
plot(Test_data(idx1,1), Test_data(idx1,2), 'r.'); hold on;
plot(Test_data(idx2,1), Test_data(idx2,2), 'g.'); hold on;
plot(Test_data(idx3,1), Test_data(idx3,2), 'b.'); hold off;