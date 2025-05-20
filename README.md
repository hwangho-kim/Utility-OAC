# HVAC 최적화 시스템 명세서

## 1. 시스템 개요

본 문서는 반도체 클린룸의 공조기(HVAC) 시스템에 대한 부하 예측 및 관리 최적화 머신러닝 알고리즘의 명세를 기술합니다. 이 시스템은 외기 조건, OAC(Out Air Controller) 유닛별 코일 목표 온도, 그리고 필요에 따라 사용자가 수동으로 지정한 코일 개도율을 입력받습니다. 이를 바탕으로 각 OAC 유닛의 코일별 개도율 및 공기 상태 변화를 예측하고, 이 예측을 기반으로 저온/고온 냉동기 시스템 및 냉각탑 시스템의 최적 운전 상태(가동 대수, 부하율, 예상 소비 전력 등)와 시스템 레벨의 주요 지표(차압, 밸브 개도율 등)를 도출합니다.

시스템은 다음과 같은 주요 기능을 포함합니다:
* **데이터 처리**: 상세 더미 데이터 생성(10분 주기) 또는 외부 CSV 파일(초당 데이터의 경우 10분 단위로 리샘플링)을 통한 학습/테스트 데이터 준비.
* **모델 학습**:
    * 14대의 OAC 유닛 각각에 대해, 5개 코일(예열, 예냉, 냉각, 재열, 가습)별로 독립적인 개도율 예측 모델(RandomForestRegressor) 학습.
    * 저온/고온 냉동기 시스템별로 주요 지표(메인 차압, 메인/서브 차압 밸브 개도율) 예측 모델(RandomForestRegressor) 학습.
* **예측 및 최적화**: 학습된 모델을 사용하여 다음을 수행:
    * 각 OAC 유닛의 코일별 개도율 및 공기 상태 변화 예측.
    * 전체 OAC 시스템의 냉각/예냉 부하 계산.
    * 계산된 부하를 바탕으로 냉동기(저온/고온) 최적 운전 조합(가동 대수, 부하율 등) 결정 및 시스템 지표 예측.
    * 냉동기 운전에 따른 총 방열 부하를 계산하고, 냉각탑 최적 운전 조합 결정.
* **상세 결과 출력**: OAC 유닛별 상세 예측 결과, 중간 계산 과정, 학습된 모델 정보, 최종 최적화 결과 등을 콘솔에 출력.

## 2. 시스템 전체 입력 및 출력 (Facade 기준)

### 2.1. 주요 입력 (`HVACSystemFacade.predict_and_optimize` 메소드)

* `outdoor_temp_c` (float): 현재 외기 온도 (섭씨).
* `outdoor_rh_percent` (float): 현재 외기 상대 습도 (%).
* `oac_set_points_by_unit` (dict): **OAC 유닛별, 각 코일 단계별 목표 온도 설정값.**
    * 외부 키: OAC 유닛 ID 문자열 (예: `'oac_0'`, `'oac_1'`, ...).
    * 내부 키: 코일 종류 문자열 (`'preheating'`, `'precooling'`, `'cooling'`, `'reheating'`, `'humidification'`).
    * 내부 값: 해당 코일의 목표 출구 온도 (섭씨).
    * 예시: `{'oac_0': {'preheating': 18.0, 'cooling': 12.0, ...}, 'oac_1': {'preheating': 18.2, 'cooling': 11.8, ...}}`
* `chilled_water_supply_set_temp` (float): 냉수 공급 설정 온도 (섭씨, 현재 로직에서는 직접적인 제어 변수보다는 참고용으로 사용될 수 있음).
* `modified_oac_coil_states_all_units` (dict, Optional): 사용자가 수동으로 변경한 특정 OAC 코일의 개도율. OAC 유닛별, 코일별로 지정 가능.
    * 외부 키: OAC 유닛 ID 문자열 (예: `'oac_0'`).
    * 내부 키: 코일 종류 문자열.
    * 내부 값: 해당 코일의 강제 설정 개도율 (0.0 ~ 1.0).
    * 예시: `{'oac_0': {'cooling': 0.8}}`

### 2.2. 주요 출력 (`HVACSystemFacade.predict_and_optimize` 메소드)

딕셔너리 형태로 반환되며, 주요 키는 다음과 같습니다:

* `oac_predictions_all_units` (dict): 시스템 내 모든 OAC 유닛 각각에 대한 코일별 예측 결과.
    * 키: OAC 유닛 ID 문자열 (예: `'oac_0'`).
    * 값 (각 OAC 유닛별 딕셔너리):
        * 키: 코일 종류 문자열.
        * 값 (각 코일별 딕셔너리):
            * `open_rate` (float): 예측된 코일 개도율.
            * `inlet_temp_c`, `inlet_rh_percent`, `inlet_h_j_kg`: 해당 코일 입구 공기 상태.
            * `outlet_temp_c`, `outlet_rh_percent`, `outlet_h_j_kg`: 해당 코일 출구 공기 상태.
            * `delta_h_j_kg` (float): 해당 코일의 엔탈피 변화량.
            * `target_set_point_temp_c` (float): 해당 코일에 설정된 목표 온도.
        * `final_outlet_temp_c`, `final_outlet_rh_percent`, `final_outlet_h_j_kg`: 해당 OAC 유닛의 최종 토출 공기 상태.
* `calculated_loads` (dict): 계산된 총 시스템 부하.
    * `total_precooling_load_watts` (float): 총 예냉 코일 부하 (Watts).
    * `total_cooling_load_watts` (float): 총 주 냉각 코일 부하 (Watts).
* `low_temp_chiller_optimization` (dict): 저온 냉동기 시스템 최적화 및 예측 결과.
    * `optimal_active_count` (int): 최적 가동 대수.
    * `active_chiller_ids` (list): 가동 냉동기 ID 리스트.
    * `load_distribution_percentage_per_active_chiller` (list): 가동 냉동기별 부하율.
    * `total_estimated_power_watts` (float): 예상 총 소비 전력 (Watts).
    * `average_load_percentage_of_active_chillers` (float): 가동 냉동기의 평균 부하율.
    * `meets_target_load_rate` (bool): 목표 부하율 달성 여부.
    * `predictions` (dict): 예측된 시스템 레벨 차압/밸브 상태 (`main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_percent`).
* `high_temp_chiller_optimization` (dict): 고온 냉동기 시스템 최적화 및 예측 결과 (저온 냉동기와 동일 구조).
* `cooling_tower_optimization` (dict): 냉각탑 최적화 결과.
    * `total_heat_rejection_load_watts` (float): 시스템 총 방열 부하 (Watts).
    * `active_tower_ids` (list): 가동 냉각탑 ID 리스트.
    * `active_tower_count` (int): 가동 냉각탑 대수.
    * `total_fan_power_watts` (float): 가동 냉각탑의 총 팬 소비 전력 (Watts).
    * `total_active_capacity_watts` (float): 가동 냉각탑의 총 방열 용량 (Watts).

## 3. 주요 모듈 및 클래스 상세

### 3.1. `EnthalpyCalculator` 클래스

* **목적**: 공기의 온도와 상대습도로부터 엔탈피를 계산하고, 엔탈피와 초기 온도로부터 새로운 온도와 상대습도를 추정합니다. (현재 구현은 단순화된 근사식을 사용합니다.)
* **주요 메소드**:
    * `calculate_enthalpy(temperature_c, relative_humidity_percent, pressure_pa=101325)`: 엔탈피 계산.
        * **입력**: `temperature_c`(섭씨 온도), `relative_humidity_percent`(상대습도 %), `pressure_pa`(대기압 Pa).
        * **출력**: 계산된 엔탈피 (J/kg 건공기).
    * `calculate_temp_humidity_from_enthalpy(enthalpy_j_kg, initial_temp_c, pressure_pa=101325)`: 엔탈피로부터 온도/습도 추정.
        * **입력**: `enthalpy_j_kg`(엔탈피 J/kg), `initial_temp_c`(초기 온도 섭씨), `pressure_pa`(대기압 Pa).
        * **출력**: 튜플 `(new_temp_c, new_humidity_percent)`.

### 3.2. `generate_dummy_data` 함수

* **목적**: 모델 학습 및 시스템 테스트를 위한 상세 시계열 더미 데이터를 지정된 주기(기본 10분)로 생성합니다. OAC 유닛별, 냉동기 유닛별, 냉각탑 유닛별 상세 데이터를 포함하며, OAC 유닛별로 약간씩 다른 Set Point를 갖도록 생성됩니다.
* **파라미터**:
    * `num_days` (int): 생성할 데이터의 일 수 (기본값 7).
    * `freq_minutes` (int): 데이터 생성 주기(분) (기본값 10).
    * `num_oac` (int): 생성할 OAC 유닛 수 (기본값 14).
    * `num_low_ch` (int): 생성할 저온 냉동기 수 (기본값 8).
    * `num_high_ch` (int): 생성할 고온 냉동기 수 (기본값 8).
    * `num_ct` (int): 생성할 냉각탑 수 (기본값 9).
* **출력**: `pandas.DataFrame`. 다음을 포함하는 테이블:
    * `datetime`: 시간 정보.
    * `outdoor_temp_c`, `outdoor_rh_percent`: 외기 온도 및 습도.
    * OAC 관련 (각 `oac_{i}` 유닛별, 각 코일별): `oac_{i}_{coil_name}_set_point_temp_c` (모든 코일 목표 온도), `oac_{i}_{coil_name}_coil_open_rate`, `oac_{i}_{coil_name}_후단온도_c`, `oac_{i}_{coil_name}_후단습도_percent` (단, `humidification` 코일의 경우 이 컬럼들은 생성되지 않음), `oac_{i}_토출온도_c`, `oac_{i}_토출노점온도_c`, `oac_{i}_토출압력_pa` (`humidification` 코일 처리 후의 상태가 이 값들로 직접 반영됨).
    * 시스템 전체 부하: `total_precooling_load_watts`, `total_cooling_load_watts`.
    * 냉동기 시스템 레벨 및 개별 냉동기 데이터, 개별 냉각탑 데이터.

### 3.3. `create_dummy_csv_second_data` 함수

* **목적**: CSV 파일 처리 기능 테스트를 위한 초당 더미 CSV 파일을 생성합니다. (실제 운영 시에는 사용되지 않음)
* **파라미터**: `filepath`(생성 파일 경로), `num_hours`(생성 데이터 시간).
* **출력**: 지정된 경로에 CSV 파일 생성.

### 3.4. `load_and_preprocess_data` 함수

* **목적**: 지정된 데이터 소스('dummy' 또는 'csv')에 따라 데이터를 로드하고, CSV인 경우 전처리(10분 단위 리샘플링, 컬럼명 매핑 시도)를 수행합니다.
* **파라미터**: `data_source`(데이터 소스 선택), `csv_filepath`(CSV 파일 경로), `num_days_dummy`(더미 데이터 일 수), `freq_minutes_dummy`(더미 데이터/리샘플링 주기).
* **출력**: `pandas.DataFrame`. 학습 및 예측에 사용될 데이터. CSV 처리 실패 시 더미 데이터 반환.

### 3.5. `OACModel` 클래스

* **목적**: 시스템 내 모든 OAC 유닛 각각에 대해, 각 코일의 개도율을 예측하는 독립적인 모델들을 학습하고, 학습된 모델 또는 규칙 기반으로 개도율 및 공기 상태 변화를 예측합니다. 각 OAC 유닛은 개별적인 목표 온도 설정값을 입력받아 사용하며, 가습 코일 후의 상태는 OAC의 최종 토출 상태입니다.
* **주요 속성**:
    * `num_oac_units_in_system` (int): 시스템 내 총 OAC 유닛 수.
    * `trained_models` (dict): `{oac_unit_id: {coil_type: RandomForestRegressor_model}}` 형태로 학습된 모델 저장.
    * `feature_names_per_coil` (dict): `{oac_unit_id: {coil_type: [feature_names]}}` 형태로 학습 시 사용된 피처 이름 저장.
    * `design_delta_h` (dict): 코일별 설계 엔탈피 변화량.
* **주요 메소드**:
    * `train_coil_models(df_train_data_full)`: `num_oac_units_in_system` 수만큼의 OAC 유닛 각각에 대해, 코일 타입별 독립 모델 학습.
    * `predict_coil_open_rates(outdoor_temp_c, outdoor_rh_percent, oac_set_points_by_unit, modified_oac_coil_states_all_units=None)`: 모든 OAC 유닛에 대해 코일별 예측 수행. `oac_set_points_by_unit`을 통해 유닛별 SP 사용.
    * `_predict_single_oac_rule_based(outdoor_temp_c, outdoor_rh_percent, oac_set_points_this_unit, modified_coil_states_this_oac=None)`: 단일 OAC 유닛 규칙 기반 예측 (내부 헬퍼).

### 3.6. `ChillerModel` 클래스

* **목적**: 냉동기 시스템(저온/고온 타입별)의 시스템 레벨 주요 운전 지표(메인 차압, 메인 차압 밸브 개도율, 서브 차압 밸브 개도율) 예측 모델 학습 및 예측값 제공.
* **주요 속성**: `trained_models`, `feature_names_chiller`.
* **주요 메소드**:
    * `train_pressure_valve_models(df_train_data)`: 모델 학습.
    * `predict_pressure_valve(current_total_load_watts, num_active_chillers_of_type, chiller_type="low_temp")`: 예측값 반환.

### 3.7. `ChillerOptimizer` 클래스

* **목적**: 주어진 냉각 부하에 대해 냉동기 최적 가동 대수 및 부하 분담률 결정.
* **주요 속성**: `chiller_specs`, `target_load_percentage`.
* **주요 메소드**:
    * `_get_cop_at_load(cop_points, load_percentage)`: 부하율에 따른 COP 계산 (내부 사용, `UnboundLocalError` 수정됨).
    * `optimize_chiller_operation(required_total_load_watts, chiller_type="low_temp")`: 최적 운전 조합 탐색.

### 3.8. `CoolingTowerOptimizer` 클래스

* **목적**: 총 방열 부하 처리를 위한 냉각탑 최적 가동 조합 결정.
* **주요 속성**: `tower_specs`, `sorted_towers`.
* **주요 메소드**:
    * `optimize_tower_operation(total_heat_rejection_load_watts)`: 최적 가동 조합 탐색.

### 3.9. `HVACSystemFacade` 클래스

* **목적**: 시스템 하위 모듈 통합 및 전체 예측/최적화 프로세스 관리.
* **주요 속성**: `oac_model`, `chiller_model`, `chiller_optimizer`, `cooling_tower_optimizer`.
* **주요 메소드**:
    * `__init__(oac_model_instance, chiller_model_instance)`: 학습된 모델 인스턴스 주입.
    * `predict_and_optimize(outdoor_temp_c, outdoor_rh_percent, oac_set_points_by_unit, chilled_water_supply_set_temp, modified_oac_coil_states_all_units=None)`:
        * **입력**: `oac_set_points_by_unit`으로 OAC 유닛별 SP 입력받음.
        * **동작**: OAC 유닛별 예측 -> 총 부하 계산 -> 냉동기 최적화 및 예측 -> 냉각탑 최적화. (내부에 중간 계산 과정 `DEBUG` print문 추가됨)
        * **출력**: 2.2. 주요 출력 참조.

## 4. 메인 실행 블록 (`if __name__ == '__main__':`)

* **목적**: 전체 시스템의 작동 흐름 시연.
* **주요 단계**:
    1.  데이터 소스 선택 (`data_source_choice`, `csv_file_path_input`) 및 `load_and_preprocess_data` 호출.
    2.  `OACModel` 및 `ChillerModel` 객체 생성 및 모델 학습.
    3.  학습된 모델 정보(타입, 파라미터, 피처 중요도 등) 확인 예시 출력.
    4.  `HVACSystemFacade` 객체 생성.
    5.  하절기 시나리오 실행:
        * OAC 유닛별로 다른 목표 온도 설정값(`oac_set_points_by_unit_summer`)을 생성하여 Facade에 전달.
        * 결과 출력: 각 OAC 유닛별 코일 상세 정보(목표 SP, 입/출구 상태, 엔탈피 변화 등) 및 모든 OAC 유닛의 개도율 요약 테이블(형식 수정됨)을 포함하여 중간 계산 과정을 상세히 표시.

