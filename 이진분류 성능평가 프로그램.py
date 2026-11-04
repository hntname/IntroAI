#이진분류 성능평가 프로그램

#혼동행렬 값
TP = float(input("Enter True Positive..."))
FP = float(input("Enter False Positive..."))
FN = float(input("Enter False Negative..."))
TN = float(input("Enter True Negaitive..."))

#전체 데이터 수 
N = TP+FP+FN+TN

#정확도
accuracy = (TP+TN)/N

#민감도, 재현율, recall
sensitivity = TP/(TP+FN)

#특이도
specificity = TN/(TN+FP)

#정밀도
precision = TP/(TP+FP)

#음성에측도
npv = TN/(FN+TN)

#F1-score
f1_score = 2*((precision*sensitivity)/(precision+sensitivity))

#오분류율
error_rate = (FP+FN)/N

print("========== 혼동행렬 ==========")
print("             실제 +    실제 -")
print(f"예측 +        {TP:5d}    {FP:5d}")
print(f"예측 -        {FN:5d}    {TN:5d}")

print("\n========== 성능 평가 ==========")
print(f"정확도 accuracy        : {accuracy:.2%}")
print(f"민감도 sensitivity     : {sensitivity:.2%}")
print(f"특이도 specificity     : {specificity:.2%}")
print(f"정밀도 precision       : {precision:.2%}")
print(f"음성예측도 NPV         : {npv:.2%}")
print(f"f1-score               : {f1_score:.2%}")
print(f"오분류율 error rate    : {error_rate:.2%}")

print("\n========== 오류 해석 ==========")
print(f"FP(False Positive) = {FP}")
print("-> 실제 음성을 양성으로 잘못 판단")
print("-> 불필요한 검사나 처치 등 '비용' 발생 가능")
print()
print(f"FN(False Negative) = {FN}")
print("-> 실제 양성을 음성으로 잘못 판단")
print("-> 질병이나 이상을 놓치는 '위험' 발생 가능")
