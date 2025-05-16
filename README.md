# HVAC 최적화 알고리즘 (`hvac_optimization_algorithm_v4`) 설명 및 입출력 변수

## 1. 개요

`CleanroomHVACOptimizer` 클래스는 반도체 클린룸의 공조기(OAC), 냉동기, 냉각탑 시스템의 운영을 최적화하기 위해 설계되었습니다. 이 알고리즘은 과거 운전 데이터를 기반으로 머신러닝 모델을 학습하고, 현재 조건에 따라 각 설비의 부하를 예측하며, 에너지 효율과 시스템 안정성을 고려하여 최적의 가동 대수를 산출합니다. 특히 OAC 코일의 개도율 예측 시, 각 코일 전후의 엔탈피 변화량(델타 엔탈피)을 중요한 학습 피처로 활용합니다.

## 2. 주요 클래스 및 메소드 설명

### `CleanroomHVACOptimizer` 클래스

-   **`__init__(self, num_oacs=14, num_low_chillers=8, num_high_chillers=8, num_cooling_towers=9)`**
    -   **설명:** 클래스 초기화 메소드. OAC, 저온/고온 냉동기, 냉각탑의 대수와 각 설비의 정격 용량, 최적 운전 부하율 범위 등 기본 파라미터를 설정합니다.
    -   모델, 스케일러, 피처명 등을 저장할 딕셔너리들을 초기화합니다.

-   **`_prepare_oac_data_single(self, df_full_input, oac_id_num)`**
    -   **설명:** 특정 OAC의 학습 데이터를 준비합니다. 외기 조건, FAN Hz, 각 코일의 설정값(set point)과 함께, **각 코일의 입구(전단) 엔탈피** 및 **코일 통과 시 엔탈피 변화량(델타 엔탈피)**을 계산하여 주요 입력 피처로 생성합니다.
    -   **주요 계산:**
        -   외기 엔탈피 계산.
        -   OAC 내 코일 처리 순서(예열 → 예냉 → 냉각 → 승온)에 따라 각 코일의 입구 엔탈피와 출구 엔탈피(코일 후단 온도/습도 데이터 기반)를 순차적으로 계산합니다.
        -   델타 엔탈피 = 출구 엔탈피 - 입구 엔탈피.
    -   **반환:** 학습용 입력 피처(X), 타겟 피처(y, 코일 개도율), 사용된 피처명 리스트, 타겟명 리스트.

-   **`train_oac_models(self, df_historical_data)`**
    -   **설명:** 모든 OAC(14대)에 대해, `_prepare_oac_data_single`를 통해 준비된 데이터를 사용하여 코일 개도율 예측 모델(Scikit-learn의 `MultiOutputRegressor`와 `RandomForestRegressor` 결합)을 학습시킵니다.
    -   학습된 모델과 데이터 스케일러(StandardScaler)를 클래스 내에 저장합니다.

-   **`predict_oac_coil_opening_ratios(self, current_outdoor_conditions_dict, oac_setpoints_list_of_dicts, modified_coil_info=None)`**
    -   **설명:** 현재 외기 조건과 OAC별 설정값을 입력받아, 학습된 모델을 사용하여 각 OAC의 5개 코일(예열, 예냉, 냉각, 승온, 가습) 개도율을 예측합니다.
    -   **중요 특징:** 학습 시에는 실제 측정된 (또는 더미 데이터에서 생성된) 전단 엔탈피와 델타 엔탈피를 피처로 사용했지만, 예측 시점에서는 이러한 값을 직접 알 수 없습니다. 따라서 이 메소드 내에서는 각 코일의 전단 엔탈피와 "목표" 델타 엔탈피를 현재 설정값과 이전 코일의 (목표) 후단 상태를 기반으로 **추정**하여 모델 입력으로 사용합니다. 이는 모델이 학습 과정에서 입력된 설정값 및 전단 엔탈피와 실제 델타 엔탈피, 그리고 최종 코일 개도율 간의 관계를 학습했다는 가정에 기반합니다. (이 추정 로직의 정확도는 실제 시스템 반응과 다를 수 있으며, 개선의 여지가 있습니다.)
    -   사용자가 특정 코일의 개도율을 수동으로 수정할 경우, 해당 값을 예측 결과에 반영합니다.

-   **`_prepare_chiller_data_common(self, df_full, chiller_type)`**
    -   **설명:** 저온 또는 고온 냉동기 모델 학습을 위한 공통 데이터 준비 로직입니다. OAC 코일 개도율(예냉 또는 냉각 코일)의 총합(FAN Hz로 가중 가능)을 주요 부하 지표로 계산하고, 외기 조건과 함께 입력 피처를 구성합니다.
    -   **반환:** 학습용 입력 피처(X), 타겟 피처(y, 냉동기 메인/서브 차압 및 개도율), 사용된 피처명 리스트, 타겟명 리스트.

-   **`train_chiller_models(self, df_historical_data)`**
    -   **설명:** 저온 및 고온 냉동기 각각에 대해, `_prepare_chiller_data_common`으로 준비된 데이터를 사용하여 운전 파라미터 예측 모델(`RandomForestRegressor` 기반)을 학습시킵니다.

-   **`predict_chiller_parameters(self, all_oac_coil_predictions_dict, current_outdoor_conditions_dict)`**
    -   **설명:** `predict_oac_coil_opening_ratios`에서 예측된 모든 OAC의 코일 개도율과 현재 외기 조건을 입력받아, 학습된 모델을 사용하여 저온/고온 냉동기의 주요 운전 파라미터(메인 차압, 메인 차압 개도율, 서브 차압 개도율)와 계산된 총 부하 지표를 예측합니다.

-   **`_calculate_optimized_units(self, total_load, capacity_per_unit, max_units, min_load_ratio, max_load_ratio)`**
    -   **설명:** (내부 헬퍼 함수) 특정 설비군(냉동기 또는 냉각탑)에 대해, 예측된 총 부하, 설비 대당 정격 용량, 최대 가용 대수, 그리고 사전에 정의된 최적 운전 부하율 범위(최소/최대)를 고려하여 가장 효율적인 가동 대수를 계산합니다.
    -   **로직:**
        1.  부하를 감당할 수 있는 이론적 최소 대수를 계산합니다.
        2.  최소 필요 대수부터 최대 가용 대수까지 각 가동 대수 시나리오를 평가합니다.
        3.  각 시나리오에서 대당 평균 부하율을 계산하고, 이 부하율이 과도하게 낮거나 높지 않으면서 "최적 운전 부하율 범위"의 중간값에 가장 가까운 대수를 선택합니다. (낮은 부하율에는 패널티를 부여하여 가급적 피하도록 유도)

-   **`optimize_chiller_operating_units(self, chiller_parameter_predictions)`**
    -   **설명:** `predict_chiller_parameters`에서 예측된 냉동기 부하 지표를 입력받아, `_calculate_optimized_units` 헬퍼 함수를 사용하여 저온 및 고온 냉동기 각각의 최적 가동 대수를 산출합니다.

-   **`optimize_cooling_tower_operating_units(self, optimal_chiller_counts_dict, chiller_parameter_predictions)`**
    -   **설명:** 최적화된 냉동기 가동 대수와 냉동기의 실제 예측 부하량을 입력받아, 냉각탑의 총 열 방출 부하를 추정합니다 (COP 가정치 사용). 이 추정된 부하를 바탕으로 `_calculate_optimized_units` 헬퍼 함수를 사용하여 냉각탑의 최적 가동 대수를 산출합니다.

## 3. 주요 입출력 변수

### `predict_oac_coil_opening_ratios`

-   **입력 (Input):**
    -   `current_outdoor_conditions_dict` (dict): 현재 외기 상태.
        -   `'외기_온도'` (float): 현재 외기 온도 (°C).
        -   `'외기_습도'` (float): 현재 외기 상대 습도 (%).
    -   `oac_setpoints_list_of_dicts` (list of dicts): 각 OAC별 설정값 리스트. 각 OAC의 딕셔너리는 다음 키를 포함합니다 (예시: OAC 1).
        -   `'OAC1_FAN_Hz'` (float): OAC 1의 팬 주파수 (Hz).
        -   `'OAC1_예열_set_point'` (float): 예열 코일 설정 온도 (°C).
        -   `'OAC1_예냉_set_point'` (float): 예냉 코일 설정 온도 (°C).
        -   `'OAC1_냉각_set_point'` (float): 냉각 코일 설정 온도 (°C).
        -   `'OAC1_승온_set_point'` (float): 승온 코일 설정 온도 (°C).
        -   `'OAC1_가습_set_point'` (float): 가습 코일 설정값 (%).
    -   `modified_coil_info` (dict, optional): 사용자가 수동으로 수정한 코일 정보.
        -   `'oac_id_num'` (int): 수정 대상 OAC 번호.
        -   `'coil_target_name'` (str): 수정 대상 코일의 전체 타겟명 (예: `'OAC1_냉각_개도율'`).
        -   `'value'` (float): 수정할 개도율 값 (%).
-   **출력 (Output):**
    -   `all_oac_predictions` (dict): OAC ID를 키로, 해당 OAC의 코일별 예측 개도율을 값(딕셔너리 형태)으로 가짐.
        -   예: `{1: {'OAC1_예열_개도율': 25.5, 'OAC1_냉각_개도율': 60.1, ...}, 2: {...}}`

### `predict_chiller_parameters`

-   **입력 (Input):**
    -   `all_oac_coil_predictions_dict` (dict): `predict_oac_coil_opening_ratios`의 반환값.
    -   `current_outdoor_conditions_dict` (dict): 현재 외기 상태 (위와 동일).
-   **출력 (Output):**
    -   `chiller_predictions_output` (dict): 저온/고온 냉동기별 예측 파라미터 및 부하 지표.
        -   `'low_temp_chiller'` (dict):
            -   `'저온냉동기_메인차압압력'` (float)
            -   `'저온냉동기_메인차압개도율'` (float)
            -   `'저온냉동기_서브차압개도율'` (float)
            -   `'calculated_total_load_indicator'` (float): 계산된 저온 냉동기 총 부하 지표.
        -   `'high_temp_chiller'` (dict): (고온 냉동기에 대한 유사한 정보)

### `optimize_chiller_operating_units`

-   **입력 (Input):**
    -   `chiller_parameter_predictions` (dict): `predict_chiller_parameters`의 반환값.
-   **출력 (Output):**
    -   `dict`: 최적 냉동기 가동 대수.
        -   `'optimal_low_temp_chillers_count'` (int): 최적 저온 냉동기 가동 대수.
        -   `'optimal_high_temp_chillers_count'` (int): 최적 고온 냉동기 가동 대수.

### `optimize_cooling_tower_operating_units`

-   **입력 (Input):**
    -   `optimal_chiller_counts_dict` (dict): `optimize_chiller_operating_units`의 반환값.
    -   `chiller_parameter_predictions` (dict): `predict_chiller_parameters`의 반환값 (냉동기 실제 부하량 참조용).
-   **출력 (Output):**
    -   `dict`: 최적 냉각탑 가동 대수.
        -   `'optimal_cooling_towers_count'` (int): 최적 냉각탑 가동 대수.

## 4. 더미 데이터 생성 로직 (`if __name__ == '__main__':`) 주요 변수

메인 실행 블록에서는 테스트 및 시연을 위한 1년치 1시간 간격 더미 데이터를 생성합니다. 주요 생성 변수는 다음과 같습니다.

-   **시간 변수:** `hour_of_day`, `day_of_year`, `month`
-   **외기 변수:** `외기_온도`, `외기_습도` (계절성 및 일일 변화 패턴 적용)
-   **OAC 변수 (OAC{i}_ 접두사):**
    -   `FAN_Hz`
    -   각 코일별 (`예열`, `예냉`, `냉각`, `승온`, `가습`) `_set_point`
    -   각 코일별 (`예열`, `예냉`, `냉각`, `승온`) `_후단_온도`, `_후단_습도` (델타 엔탈피 계산의 기반)
    -   각 코일별 (`예열`, `예냉`, `냉각`, `승온`, `가습`) `_개도율` (외기 조건 및 다른 요인에 따른 경향성 부여)
-   **냉동기 변수 (저온냉동기_ 또는 고온냉동기_ 접두사):**
    -   `메인차압개도율`, `서브차압개도율`, `메인차압압력` (OAC 부하 지표에 연동)
    -   `메인supply압력`, `메인supply온도`, `메인return압력`, `메인return온도`

이 더미 데이터는 `_prepare_oac_data_single` 및 `_prepare_chiller_data_common` 함수를 통해 모델 학습에 적합한 형태로 가공됩니다.
