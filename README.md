# HVAC 최적화 시스템 명세서 

## 1. 시스템 개요

본 문서는 반도체 클린룸의 공조기(HVAC) 시스템에 대한 부하 예측 및 관리 최적화 머신러닝 알고리즘의 명세를 기술합니다. 시스템은 외기 조건 및 설비 설정값을 입력받아 OAC(Out Air Controller) 코일 개도율, 냉동기(저온/고온) 및 냉각탑의 최적 운전 상태를 예측하고 제안합니다.

주요 특징은 다음과 같습니다:
* 상세 더미 데이터 생성을 통해 다양한 운전 시나리오 모사.
* 14대의 OAC 유닛 각각에 대해 독립적인 코일별 머신러닝 모델 학습 및 예측 수행.
* 학습된 모델을 사용하여 시스템 전체의 부하 예측 및 에너지 최적화 시뮬레이션 기능 제공.

## 2. 시스템 전체 입력 및 출력

### 2.1. 주요 입력 (`HVACSystemFacade.predict_and_optimize` 기준)

* `outdoor_temp_c` (float): 현재 외기 온도 (섭씨)
* `outdoor_rh_percent` (float): 현재 외기 상대 습도 (%)
* `oac_set_points_all_coils` (dict): OAC 각 코일 단계별 **목표 온도** 설정값. 이 설정은 시스템 내 모든 OAC 유닛에 공통으로 적용됩니다.
    * 키: 코일 종류 문자열 (`'preheating'`, `'precooling'`, `'cooling'`, `'reheating'`, `'humidification'`)
    * 값: 해당 코일의 목표 출구 온도 (섭씨)
    * 예: `{'preheating': 18.0, 'precooling': 20.0, 'cooling': 12.0, 'reheating': 23.0, 'humidification': 23.5}`
* `chilled_water_supply_set_temp` (float): 냉수 공급 설정 온도 (섭씨, 현재 로직에서는 직접적인 제어 변수보다는 참고용으로 사용될 수 있음)
* `modified_oac_coil_states_all_units` (dict, Optional): 사용자가 수동으로 변경한 특정 OAC 코일의 개도율. OAC 유닛별, 코일별로 지정 가능.
    * 외부 키: OAC 유닛 ID 문자열 (예: `'oac_0'`, `'oac_1'`, ...)
    * 내부 키: 코일 종류 문자열
    * 내부 값: 해당 코일의 강제 설정 개도율 (0.0 ~ 1.0)
    * 예: `{'oac_0': {'cooling': 0.8}, 'oac_3': {'reheating': 0.5}}`

### 2.2. 주요 출력 (`HVACSystemFacade.predict_and_optimize` 기준)

딕셔너리 형태로 반환되며, 주요 키는 다음과 같습니다:

* `oac_predictions_all_units` (dict): **시스템 내 모든 OAC 유닛(14대) 각각에 대한 코일별 예측 결과.**
    * 키: OAC 유닛 ID 문자열 (예: `'oac_0'`, `'oac_1'`, ...)
    * 값 (각 OAC 유닛별 딕셔너리):
        * 키: 코일 종류 문자열 (`'preheating'`, `'precooling'`, 등)
        * 값 (각 코일별 딕셔너리):
            * `open_rate` (float): 예측된 코일 개도율 (0.0 ~ 1.0)
            * `inlet_temp_c`, `inlet_rh_percent`, `inlet_h_j_kg`: 해당 코일 입구 공기 상태
            * `outlet_temp_c`, `outlet_rh_percent`, `outlet_h_j_kg`: 해당 코일 출구 공기 상태 (단, `humidification` 코일의 경우 이 값들이 해당 OAC의 최종 토출 상태와 동일)
            * `delta_h_j_kg` (float): 해당 코일을 통과하며 발생한 엔탈피 변화량 (J/kg)
            * `target_set_point_temp_c` (float): 해당 코일에 설정된 목표 온도
        * `final_outlet_temp_c`, `final_outlet_rh_percent`, `final_outlet_h_j_kg`: 해당 OAC 유닛의 최종 토출 공기 상태
* `calculated_loads` (dict): 계산된 총 시스템 부하 (모든 OAC 유닛의 부하 합산).
    * `total_precooling_load_watts` (float): 모든 OAC의 예냉 코일에서 발생한 총 부하량 (Watts)
    * `total_cooling_load_watts` (float): 모든 OAC의 주 냉각 코일에서 발생한 총 부하량 (Watts)
* `low_temp_chiller_optimization` (dict): 저온 냉동기 시스템 최적화 및 예측 결과. (구조는 V3 명세와 동일)
* `high_temp_chiller_optimization` (dict): 고온 냉동기 시스템 최적화 및 예측 결과. (구조는 V3 명세와 동일)
* `cooling_tower_optimization` (dict): 냉각탑 최적화 결과. (구조는 V3 명세와 동일)

## 3. 주요 모듈 및 클래스 상세

### 3.1. `EnthalpyCalculator` 클래스

* **목적**: 공기의 온도와 습도로부터 엔탈피를 계산하거나, 엔탈피와 초기 온도로부터 새로운 온도와 습도를 추정합니다. (현재 구현은 매우 단순화된 Placeholder임)
* **주요 메소드**:
    * `calculate_enthalpy(temperature_c, relative_humidity_percent, pressure_pa=101325)`
    * `calculate_temp_humidity_from_enthalpy(enthalpy_j_kg, initial_temp_c, pressure_pa=101325)`

### 3.2. `generate_dummy_data` 함수

* **목적**: 모델 학습 및 시스템 테스트를 위한 상세 시계열 더미 데이터를 생성합니다. OAC 유닛별, 냉동기 유닛별, 냉각탑 유닛별 상세 데이터를 포함합니다.
* **파라미터**: `num_rows`, `num_oac`, `num_low_ch`, `num_high_ch`, `num_ct`
* **출력**: `pandas.DataFrame`. 다음을 포함하는 테이블:
    * `datetime`: 시간 정보
    * `outdoor_temp_c`, `outdoor_rh_percent`: 외기 온도 및 습도
    * OAC 관련 (각 `oac_{i}` 유닛별, 각 코일별):
        * `oac_{i}_{coil_name}_set_point_temp_c` (모든 코일 목표 온도)
        * `oac_{i}_{coil_name}_coil_open_rate`
        * `oac_{i}_{coil_name}_후단온도_c`, `oac_{i}_{coil_name}_후단습도_percent` (**단, `humidification` 코일의 경우 이 컬럼들은 생성되지 않음.**)
        * `oac_{i}_토출온도_c`, `oac_{i}_토출노점온도_c`, `oac_{i}_토출압력_pa` (**`humidification` 코일 처리 후의 상태가 이 값들로 직접 반영됨.**)
    * 시스템 전체 부하: `total_precooling_load_watts`, `total_cooling_load_watts`
    * 냉동기 시스템 레벨 및 개별 냉동기 데이터, 개별 냉각탑 데이터 (V3 명세와 유사)

### 3.3. `OACModel` 클래스

* **목적**: 시스템 내 모든 OAC 유닛 각각에 대해, 각 코일의 개도율을 예측하는 독립적인 모델들을 학습하고, 학습된 모델 또는 규칙 기반으로 개도율 및 공기 상태 변화를 예측합니다. 모든 코일의 설정값은 목표 온도를 기준으로 하며, 가습 코일 후의 상태는 OAC의 최종 토출 상태입니다.
* **주요 속성**:
    * `num_oac_units_in_system` (int)
    * `trained_models` (dict): `{oac_unit_id: {coil_type: model}}`
    * `feature_names_per_coil` (dict): `{oac_unit_id: {coil_type: [feature_names]}}`
    * `design_delta_h` (dict)
* **주요 메소드**:
    * `train_coil_models(df_train_data_full)`
        * **동작**: 각 OAC 유닛별, 각 코일 타입별 모델 학습.
            * `humidification` 코일 이전 코일(`reheating`)의 후단 상태를 특징으로 사용.
            * `humidification` 코일의 개도율 학습 시, 목표는 해당 OAC의 최종 `토출온도_c` (더미 데이터의 `oac_{i}_토출온도_c`)를 달성하는 것입니다. (실제로는 `oac_{i}_humidification_set_point_temp_c`를 목표로 함)
    * `predict_coil_open_rates(outdoor_temp_c, outdoor_rh_percent, oac_set_points_all_coils, modified_oac_coil_states_all_units=None)`
        * **동작**: 모든 OAC 유닛에 대해 코일별 예측 수행.
            * `humidification` 코일 처리 후의 공기 상태가 해당 OAC 유닛의 `final_outlet_temp_c`, `final_outlet_rh_percent`, `final_outlet_h_j_kg`가 됩니다.
        * **출력**: `dict`, 키는 OAC 유닛 ID, 값은 해당 유닛의 코일별 상세 예측 결과 및 최종 토출 상태.
    * `_predict_single_oac_rule_based(...)`
        * **동작**: 단일 OAC 유닛에 대한 규칙 기반 예측 로직. 가습 코일 처리 후 상태가 최종 토출 상태가 됨.

### 3.4. `ChillerModel` 클래스

* **목적**: 냉동기 시스템(저온/고온 타입별)의 시스템 레벨 주요 운전 지표(메인 차압, 메인 차압 밸브 개도율, 서브 차압 밸브 개도율)를 예측하는 모델을 학습하고, 이를 이용해 예측값을 제공합니다.
* **주요 속성**: `trained_models`, `feature_names_chiller`
* **주요 메소드**:
    * `train_pressure_valve_models(df_train_data)`
    * `predict_pressure_valve(current_total_load_watts, num_active_chillers_of_type, chiller_type="low_temp")`

### 3.5. `ChillerOptimizer` 클래스

* **목적**: 주어진 냉각 부하에 대해 저온 또는 고온 냉동기들의 최적 가동 대수 및 각 냉동기별 부하 분담률을 결정합니다.
* **주요 속성**: `chiller_specs`, `target_load_percentage`
* **주요 메소드**:
    * `_get_cop_at_load(cop_points, load_percentage)`
    * `optimize_chiller_operation(required_total_load_watts, chiller_type="low_temp")`

### 3.6. `CoolingTowerOptimizer` 클래스

* **목적**: 냉동기에서 발생하는 총 방열 부하를 처리하기 위한 냉각탑의 최적 가동 조합을 결정합니다.
* **주요 속성**: `tower_specs`, `sorted_towers`
* **주요 메소드**:
    * `optimize_tower_operation(total_heat_rejection_load_watts)`

### 3.7. `HVACSystemFacade` 클래스

* **목적**: 시스템의 모든 하위 모듈을 통합하고 전체 예측 및 최적화 프로세스를 관리하는 인터페이스.
* **주요 속성**: `oac_model`, `chiller_model`, `chiller_optimizer`, `cooling_tower_optimizer`
* **주요 메소드**:
    * `__init__(oac_model_instance, chiller_model_instance)`
    * `predict_and_optimize(outdoor_temp_c, outdoor_rh_percent, oac_set_points_all_coils, chilled_water_supply_set_temp, modified_oac_coil_states_all_units=None)`
        * **동작**:
            1.  `OACModel`을 사용하여 모든 OAC 유닛 각각의 코일별 상태 예측.
            2.  모든 OAC 유닛들의 예냉/냉각 코일 부하를 합산하여 총 시스템 부하 계산.
            3.  (이하 V3 명세와 동일)
        * **출력**: (2.2. 주요 출력 (Facade 기준) 참조)

## 4. 메인 실행 블록 (`if __name__ == '__main__':`)

* **목적**: 전체 시스템의 작동 흐름을 시연합니다.
* **주요 단계**:
    1.  `generate_dummy_data` 함수 호출 (상세 데이터 생성, 가습 코일 후단 데이터는 생성 안 함).
    2.  `OACModel` 객체 생성 및 `train_coil_models` 호출 (OAC 유닛별 모델 학습).
    3.  `ChillerModel` 객체 생성 및 `train_pressure_valve_models` 호출.
    4.  학습된 모델 인스턴스들을 `HVACSystemFacade` 생성자에 주입.
    5.  하절기 시나리오 실행 및 결과 출력 (OAC 유닛별 코일 개도율 및 최종 토출 상태 테이블 형식 표시).

