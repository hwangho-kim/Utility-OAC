# HVAC 최적화 알고리즘 (`hvac_optimization_algorithm`) 상세 설명 및 입출력 변수

## 1. 개요

`CleanroomHVACOptimizer` 클래스는 반도체 클린룸의 공조기(OAC), 냉동기, 냉각탑으로 구성된 HVAC 시스템의 운영을 최적화하기 위해 개발되었습니다. 이 알고리즘의 핵심 목표는 과거 운전 데이터를 기반으로 머신러닝 모델을 학습시켜, 현재의 외기 조건 및 사용자의 OAC 설정값에 따라 각 설비의 부하를 정확하게 예측하고, 에너지 효율성과 시스템 안정성을 종합적으로 고려하여 최적의 설비 가동 대수를 산출하는 것입니다.

특히, OAC 코일의 개도율을 예측하는 모델에서는 각 코일 전후의 **엔탈피 변화량(델타 엔탈피)**을 중요한 학습 피처로 활용하여 물리적 현상에 기반한 예측 정확도 향상을 추구합니다. 또한, 냉동기 및 냉각탑의 최적 가동 대수 결정 시, 단순 부하 만족을 넘어 정의된 **최적 운전 부하율 범위** 내에서 설비가 운영되도록 유도하여 효율적인 부하 분담을 목표로 합니다.

알고리즘은 Python으로 구현되었으며, Scikit-learn 라이브러리를 활용하여 머신러닝 모델(주로 `RandomForestRegressor`)을 구축하고 학습합니다.

## 2. 주요 클래스 및 메소드 상세 설명

### `CleanroomHVACOptimizer` 클래스

-   **`__init__(self, num_oacs=14, num_low_chillers=8, num_high_chillers=8, num_cooling_towers=9)`**
    -   **설명:** 클래스 생성자. HVAC 시스템의 기본 구성(각 설비의 대수)과 최적화 로직에 사용될 주요 파라미터(설비별 정격 용량, 최적 운전 부하율 범위 등)를 초기화합니다. 또한, 학습된 모델, 데이터 스케일러, 피처 및 타겟 이름 등을 저장할 내부 딕셔너리 변수들을 준비합니다.
    -   **주요 내부 변수 초기화:**
        -   `rated_capacity_per_low_chiller`, `rated_capacity_per_high_chiller`, `rated_heat_rejection_per_ct`: 각 설비의 대당 정격 처리 용량 (최적화 시 기준값).
        -   `optimal_load_ratio_min`, `optimal_load_ratio_max`: 설비가 효율적으로 운전될 수 있다고 가정하는 부하율 범위.

-   **`_prepare_oac_data_single(self, df_full_input, oac_id_num)`**
    -   **설명:** (내부 헬퍼 함수) 특정 OAC(ID `oac_id_num`)의 머신러닝 모델 학습을 위한 데이터를 준비합니다. 이 함수는 OAC 코일 개도율 예측의 핵심인 **델타 엔탈피 피처 생성** 로직을 포함합니다.
    -   **주요 로직:**
        1.  **외기 엔탈피 계산:** 입력된 데이터프레임에 외기 온도/습도만 있다면, 이를 이용해 외기 엔탈피를 계산합니다.
        2.  **코일 단계별 엔탈피 계산:** OAC 내부 공기 처리 순서(예열 → 예냉 → 냉각 → 승온)에 따라 각 코일의 입구(전단) 엔탈피와 출구(후단) 엔탈피를 순차적으로 계산합니다.
            -   첫 코일(예열)의 입구 엔탈피는 외기 엔탈피입니다.
            -   이후 코일의 입구 엔탈피는 바로 이전 코일의 출구 엔탈피가 됩니다.
            -   각 코일의 출구 엔탈피는 해당 코일의 **후단 온도 및 후단 습도 데이터** (과거 데이터에 존재해야 함)를 사용하여 계산합니다.
        3.  **델타 엔탈피($\Delta h$) 계산:** 각 코일별로 `출구 엔탈피 - 입구 엔탈피`를 계산하여 $\Delta h$ 값을 도출합니다.
        4.  **피처 구성:** 최종적으로 OAC 모델 학습에 사용될 입력 피처(X)는 다음과 같습니다:
            -   기본 외기 조건 (온도, 습도, 엔탈피)
            -   해당 OAC의 FAN Hz
            -   각 코일(예열, 예냉, 냉각, 승온)의 설정 온도(set point)
            -   각 코일(예열, 예냉, 냉각, 승온)의 **계산된 전단 엔탈피**
            -   각 코일(예열, 예냉, 냉각, 승온)의 **계산된 델타 엔탈피**
            -   가습 코일의 설정값 (가습 코일은 델타 엔탈피 계산에서 제외)
        5.  타겟 피처(y)는 각 코일의 실제 개도율입니다.
    -   **반환:** 준비된 학습용 입력 피처(X), 타겟 피처(y), 그리고 사용된 피처 및 타겟의 이름 리스트. 데이터가 불충분하면 `None`을 반환합니다.

-   **`train_oac_models(self, df_historical_data)`**
    -   **설명:** 모든 OAC(기본 14대)에 대해 코일 개도율 예측 모델을 학습시킵니다. 각 OAC별로 `_prepare_oac_data_single` 함수를 호출하여 학습 데이터를 준비하고, `MultiOutputRegressor`와 `RandomForestRegressor`를 결합한 모델을 학습시켜 내부적으로 저장합니다. 입력 데이터 스케일링을 위해 `StandardScaler`도 함께 사용 및 저장합니다.
    -   **[개선 제안 주석 반영]:** RandomForest 외 다른 모델(예: Gradient Boosting, Neural Network) 검토 및 하이퍼파라미터 최적화 필요.

-   **`predict_oac_coil_opening_ratios(self, current_outdoor_conditions_dict, oac_setpoints_list_of_dicts, modified_coil_info=None)`**
    -   **설명:** 현재의 실시간 외기 조건과 각 OAC의 설정값을 입력받아, 이미 학습된 OAC 모델들을 사용하여 각 OAC의 5가지 코일(예열, 예냉, 냉각, 승온, 가습)에 대한 개도율을 예측합니다.
    -   **델타 엔탈피 처리 (예측 시):** 학습 시에는 실제 과거 데이터의 후단 온/습도를 기반으로 정확한 전단 엔탈피와 델타 엔탈피를 계산하여 피처로 사용했지만, 예측 시점에서는 코일의 실제 후단 상태(즉, 실제 델타 엔탈피)를 미리 알 수 없습니다. 이 문제를 해결하기 위해, 이 메소드 내에서는 각 코일의 **전단 엔탈피**와 **"목표" 델타 엔탈피**를 현재 주어진 설정값(set point)과 이전 코일의 "목표" 후단 상태(set point 기반으로 추정된 엔탈피)를 이용하여 **근사적으로 추정**합니다. 이렇게 추정된 값들을 모델의 입력 피처로 사용하여 개도율을 예측합니다. 모델은 학습 과정에서 이러한 입력(설정값, 추정된 전단/목표 델타 엔탈피)과 실제 코일 개도율 간의 복잡한 관계를 학습했다고 가정합니다.
    -   **[개선 제안 주석 반영]:**
        1.  예측 시 델타 엔탈피 추정 로직의 정교화 (예: 습도 변화를 보다 정확히 모델링, 또는 반복적 예측을 통해 후단 상태를 점진적으로 추정하는 방식).
        2.  델타 엔탈피 피처 없이, set_point, 전단 엔탈피(외기 또는 이전 코일 SP 기반 추정), 외기 조건만으로 학습/예측하는 모델 구조로 변경하는 것을 고려 (피처 일관성 확보).
        3.  한 코일의 개도율을 수동으로 수정했을 때, 다른 코일들의 개도율이 물리적/제어적 관계에 따라 자동으로 연동되어 조정되는 로직 추가 (현재는 단순 덮어쓰기).

-   **`_prepare_chiller_data_common(self, df_full, chiller_type)`**
    -   **설명:** (내부 헬퍼 함수) 저온 또는 고온 냉동기 모델의 학습 데이터를 준비합니다. 핵심 로직은 모든 OAC의 관련 코일(저온 냉동기의 경우 냉각 코일, 고온 냉동기의 경우 예냉 코일)의 개도율 총합을 계산하여 (필요시 FAN Hz로 가중치 부여) 해당 냉동기군의 부하를 나타내는 주요 지표(`Total_{chiller_type}_Coil_Load_Indicator`)로 사용하는 것입니다. 이 부하 지표와 외기 조건을 입력 피처로 구성합니다.
    -   **반환:** 학습용 입력 피처(X), 타겟 피처(y: 해당 냉동기군의 메인 차압, 메인 차압 개도율, 서브 차압 개도율), 그리고 사용된 피처 및 타겟의 이름 리스트.

-   **`train_chiller_models(self, df_historical_data)`**
    -   **설명:** 저온 냉동기군과 고온 냉동기군 각각에 대해, 운전 파라미터(메인/서브 차압 및 개도율) 예측 모델을 학습시킵니다. `_prepare_chiller_data_common` 함수로 데이터를 준비하고, `RandomForestRegressor` 기반의 모델을 학습합니다.

-   **`predict_chiller_parameters(self, all_oac_coil_predictions_dict, current_outdoor_conditions_dict)`**
    -   **설명:** `predict_oac_coil_opening_ratios`로부터 얻은 모든 OAC의 예측된 코일 개도율과 현재 외기 조건을 입력으로 받아, 학습된 냉동기 모델을 사용하여 각 냉동기군(저온/고온)의 주요 운전 파라미터(메인 차압, 메인 차압 개도율, 서브 차압 개도율)와 계산된 총 부하 지표를 예측합니다.
    -   **[개선 제안 주석 반영]:** 냉동기 부하 지표 계산 시, OAC의 FAN_Hz 값을 가중치로 활용하면 정확도 향상에 도움이 될 수 있습니다. (현재 코드에서는 FAN_Hz 없이 단순 개도율 합산으로 되어 있어, 이 부분은 개선이 필요합니다.)

-   **`_calculate_optimized_units(self, total_load, capacity_per_unit, max_units, min_load_ratio, max_load_ratio)`**
    -   **설명:** (내부 헬퍼 함수) 특정 설비군(냉동기 또는 냉각탑)에 대한 최적 가동 대수를 계산하는 핵심 로직입니다. 예측된 총 부하, 설비 대당 정격 용량, 최대 가용 대수, 그리고 사전에 정의된 최적 운전 부하율 범위(최소/최대)를 입력받습니다.
    -   **최적화 로직:**
        1.  먼저, 주어진 총 부하를 감당하기 위한 이론적인 최소 필요 대수를 계산합니다.
        2.  이 최소 필요 대수부터 시작하여 최대 가용 대수까지, 가능한 모든 가동 대수 시나리오를 평가합니다.
        3.  각 시나리오에서 설비 대당 평균 부하율을 계산합니다.
        4.  이 평균 부하율이 과도하게 낮거나(예: 5% 미만, 매우 비효율적) 높지 않으면서(예: 100% 초과, 과부하), 사전에 정의된 "최적 운전 부하율 범위"의 중간값에 가장 가까운 점수를 가진 대수를 최적 대수로 선택합니다. 평균 부하율이 최소 최적 부하율보다 낮은 경우에는 패널티를 부여하여 해당 시나리오의 선호도를 낮춥니다. 점수가 동일할 경우, 더 적은 대수를 우선적으로 선택합니다.
    -   **[개선 제안 주석 반영]:**
        1.  실제 설비별 부분 부하 효율(Part Load Efficiency) 곡선을 반영하여 에너지 소비 최소화를 목표로 최적화.
        2.  잦은 기동/정지 시 발생하는 에너지 손실 및 설비 수명 단축을 고려한 비용 함수 도입.
        3.  최소 가동/정지 시간, 순차 기동/정지 로직(Stage Control) 등 실제 운영 제약 조건 추가.
        4.  강화학습(Reinforcement Learning) 등 더 고급 최적화 기법 적용 고려.

-   **`optimize_chiller_operating_units(self, chiller_parameter_predictions)`**
    -   **설명:** `predict_chiller_parameters`에서 예측된 냉동기 부하 지표를 사용하여, `_calculate_optimized_units` 헬퍼 함수를 호출함으로써 저온 및 고온 냉동기 각각의 최적 가동 대수를 산출합니다.

-   **`optimize_cooling_tower_operating_units(self, optimal_chiller_counts_dict, chiller_parameter_predictions)`**
    -   **설명:** 최적화된 냉동기 가동 대수와 냉동기의 실제 예측 부하량을 입력으로 받아, 냉각탑이 처리해야 할 총 열 방출 부하를 추정합니다. 이 때, 냉동기의 평균 성능계수(COP, 현재 코드에서는 4.0으로 가정)를 사용하여 $Q_{reject} = Q_{evap} \cdot (1 + 1/COP)$ 공식으로 열 방출량을 계산합니다. 이렇게 추정된 총 열 방출 부하를 바탕으로 `_calculate_optimized_units` 헬퍼 함수를 사용하여 냉각탑의 최적 가동 대수를 산출합니다.
    -   **[개선 제안 주석 반영]:** 냉동기 COP를 고정값 대신, 냉동기 종류, 현재 부하율, 외기 조건 등에 따라 변하는 동적 COP 모델을 사용하거나, 실제 전력 소비량 데이터를 기반으로 열 방출량을 추정하면 정확도가 향상됩니다.

## 3. 주요 입출력 변수 상세

### `predict_oac_coil_opening_ratios` (OAC 코일 개도율 예측)

-   **입력 (Input):**
    -   `current_outdoor_conditions_dict` (dict): 현재 외기 상태.
        -   `'외기_온도'` (float): 현재 외기 온도 (°C).
        -   `'외기_습도'` (float): 현재 외기 상대 습도 (%).
    -   `oac_setpoints_list_of_dicts` (list of dicts): 각 OAC(14대)별 설정값 리스트. 각 OAC의 딕셔너리는 다음 키들을 포함합니다 (예시: `OAC{i}_` 접두사 사용).
        -   `'OAC{i}_FAN_Hz'` (float): 해당 OAC의 팬 주파수 (Hz).
        -   `'OAC{i}_예열_set_point'` (float): 예열 코일 설정 온도 (°C).
        -   `'OAC{i}_예냉_set_point'` (float): 예냉 코일 설정 온도 (°C).
        -   `'OAC{i}_냉각_set_point'` (float): 냉각 코일 설정 온도 (°C).
        -   `'OAC{i}_승온_set_point'` (float): 승온 코일 설정 온도 (°C).
        -   `'OAC{i}_가습_set_point'` (float): 가습 코일 설정값 (더미 데이터에서는 %RH로 가정).
    -   `modified_coil_info` (dict, optional): 사용자가 수동으로 특정 코일의 개도율을 수정했을 경우 전달되는 정보.
        -   `'oac_id_num'` (int): 수정 대상 OAC의 ID 번호 (1~14).
        -   `'coil_target_name'` (str): 수정 대상 코일의 전체 타겟명 (예: `'OAC1_냉각_개도율'`).
        -   `'value'` (float): 사용자가 입력한 새로운 개도율 값 (%).
-   **출력 (Output):**
    -   `all_oac_predictions` (dict): OAC ID(정수)를 키로 하고, 해당 OAC의 코일별 예측 개도율을 값(딕셔너리 형태)으로 가지는 딕셔너리.
        -   내부 딕셔너리 예시 (OAC 1): `{'OAC1_예열_개도율': 25.5, 'OAC1_예냉_개도율': 30.0, ...}`

### `predict_chiller_parameters` (냉동기 운전 파라미터 예측)

-   **입력 (Input):**
    -   `all_oac_coil_predictions_dict` (dict): `predict_oac_coil_opening_ratios` 메소드의 반환값 (모든 OAC의 예측된 코일 개도율 정보).
    -   `current_outdoor_conditions_dict` (dict): 현재 외기 상태 (위와 동일).
-   **출력 (Output):**
    -   `chiller_predictions_output` (dict): 저온 및 고온 냉동기군 각각에 대한 예측된 운전 파라미터와 계산된 부하 지표를 포함하는 딕셔너리.
        -   `'low_temp_chiller'` (dict): 저온 냉동기군 정보.
            -   `'저온냉동기_메인차압압력'` (float)
            -   `'저온냉동기_메인차압개도율'` (float)
            -   `'저온냉동기_서브차압개도율'` (float)
            -   `'calculated_total_load_indicator'` (float): OAC 냉각 코일들의 총 부하로부터 계산된 저온 냉동기군의 상대적 부하 지표.
        -   `'high_temp_chiller'` (dict): 고온 냉동기군 정보 (위와 유사한 구조).

### `optimize_chiller_operating_units` (냉동기 최적 가동 대수 산출)

-   **입력 (Input):**
    -   `chiller_parameter_predictions` (dict): `predict_chiller_parameters` 메소드의 반환값.
-   **출력 (Output):**
    -   `dict`: 최적으로 판단된 냉동기 가동 대수 정보.
        -   `'optimal_low_temp_chillers_count'` (int): 최적 저온 냉동기 가동 대수.
        -   `'optimal_high_temp_chillers_count'` (int): 최적 고온 냉동기 가동 대수.

### `optimize_cooling_tower_operating_units` (냉각탑 최적 가동 대수 산출)

-   **입력 (Input):**
    -   `optimal_chiller_counts_dict` (dict): `optimize_chiller_operating_units` 메소드의 반환값 (최적 냉동기 가동 대수 정보).
    -   `chiller_parameter_predictions` (dict): `predict_chiller_parameters` 메소드의 반환값 (냉동기의 실제 예측 부하량 참조용).
-   **출력 (Output):**
    -   `dict`: 최적으로 판단된 냉각탑 가동 대수 정보.
        -   `'optimal_cooling_towers_count'` (int): 최적 냉각탑 가동 대수.

## 4. 더미 데이터 생성 로직 (`if __name__ == '__main__':`)

메인 실행 블록(`if __name__ == '__main__':`)에는 알고리즘의 테스트 및 시연을 위해 1년치 1시간 간격의 현실성 있는 더미 데이터를 생성하는 코드가 포함되어 있습니다. 이 더미 데이터는 다음과 같은 특징을 가집니다:

-   **시간적 패턴:** 외기 온도 및 습도에 계절적 변화(연간 주기)와 일일 변화(낮/밤 주기)를 사인파 형태로 적용합니다.
-   **설비 간 연관성 모방:**
    -   OAC 코일의 설정값(set point) 및 개도율이 외기 조건(특히 온도)에 따라 어느 정도 반응하도록 생성합니다.
    -   냉동기의 부하 관련 지표(예: 메인 차압 개도율)가 연결된 OAC 코일(예냉/냉각)의 총 부하 지표에 민감하게 반응하도록 생성합니다.
-   **데이터 값 범위 및 노이즈:** 각 피처의 값 범위를 현실적으로 조정하고, 약간의 랜덤 노이즈를 추가하여 실제 센서 데이터와 유사하게 만듭니다.
-   **델타 엔탈피 계산 기반:** OAC 코일별 후단 온도 및 후단 습도 데이터를 포함하여, `_prepare_oac_data_single` 함수에서 델타 엔탈피를 계산하고 이를 모델 학습에 사용할 수 있도록 합니다.

이 더미 데이터는 실제 데이터가 없을 경우 알고리즘의 작동 방식을 이해하고 검증하는 데 사용될 수 있지만, 실제 시스템에 적용하기 위해서는 반드시 해당 시스템의 과거 운전 데이터로 모델을 학습시켜야 합니다.
