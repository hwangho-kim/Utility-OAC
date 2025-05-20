# HVAC 최적화 시스템 명세서

## 1. 시스템 개요

본 문서는 반도체 클린룸의 공조기(HVAC) 시스템에 대한 부하 예측 및 관리 최적화 머신러닝 알고리즘의 명세를 기술합니다. 시스템은 외기 조건 및 설비 설정값을 입력받아 OAC(Out Air Controller) 코일 개도율, 냉동기(저온/고온) 및 냉각탑의 최적 운전 상태를 예측하고 제안합니다. 시스템은 더미 데이터 생성, 모델 학습, 그리고 학습된 모델을 사용한 시뮬레이션 기능을 포함합니다.

## 2. 시스템 전체 입력 및 출력

### 2.1. 주요 입력 (Facade 기준)

`HVACSystemFacade.predict_and_optimize` 메소드의 주요 입력:

* `outdoor_temp_c` (float): 현재 외기 온도 (섭씨)
* `outdoor_rh_percent` (float): 현재 외기 상대 습도 (%)
* `oac_set_points` (dict): OAC 각 코일 단계별 설정값.
    * 예: `{'preheating': 18.0, 'precooling': 20.0, 'cooling': 12.0, 'reheating': 23.0, 'humidification_rate': 0.0}`
    * 온도 설정 코일: 목표 출구 온도 (섭씨)
    * 가습 코일 (`humidification_rate`): 목표 개도율 (0.0 ~ 1.0)
* `chilled_water_supply_set_temp` (float): 냉수 공급 설정 온도 (섭씨, 현재 로직에서는 참고용으로 직접 사용되지 않으나 최적화 시 고려 가능)
* `modified_oac_coil_states` (dict, Optional): 사용자가 수동으로 변경한 특정 OAC 코일의 개도율.
    * 예: `{'cooling': 0.8}` (냉각 코일 개도율 80%로 강제)

### 2.2. 주요 출력 (Facade 기준)

`HVACSystemFacade.predict_and_optimize` 메소드의 주요 출력 (딕셔너리 형태):

* `oac_predictions_per_unit` (dict): 단일 대표 OAC 유닛에 대한 코일별 예측 결과.
    * 각 코일별: `open_rate` (예측 개도율), `inlet_temp_c`, `inlet_rh_percent`, `inlet_h_j_kg`, `outlet_temp_c`, `outlet_rh_percent`, `outlet_h_j_kg`, `delta_h_j_kg` (엔탈피 변화량), `target_outlet_temp_c` (목표 설정값)
    * `final_outlet_temp_c`, `final_outlet_rh_percent`, `final_outlet_h_j_kg`: OAC 최종 토출 공기 상태
* `calculated_loads` (dict): 계산된 총 시스템 부하.
    * `total_precooling_load_watts`: 총 예냉 코일 부하 (고온 냉동기 담당, Watts)
    * `total_cooling_load_watts`: 총 주 냉각 코일 부하 (저온 냉동기 담당, Watts)
* `low_temp_chiller_optimization` (dict): 저온 냉동기 최적화 및 예측 결과.
    * `optimal_active_count`: 최적 가동 대수
    * `active_chiller_ids`: 가동 냉동기 ID 리스트
    * `load_distribution_percentage_per_active_chiller`: 가동 냉동기별 부하율 리스트
    * `total_estimated_power_watts`: 예상 총 소비 전력 (Watts)
    * `average_load_percentage_of_active_chillers`: 가동 냉동기의 평균 부하율
    * `meets_target_load_rate`: 목표 부하율(55%) 달성 여부 (boolean)
    * `predictions` (dict): 예측된 시스템 레벨 차압/밸브 상태 (`main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_percent`)
* `high_temp_chiller_optimization` (dict): 고온 냉동기 최적화 및 예측 결과 (저온 냉동기와 동일한 구조)
* `cooling_tower_optimization` (dict): 냉각탑 최적화 결과.
    * `total_heat_rejection_load_watts`: 총 방열 부하 (Watts)
    * `active_tower_ids`: 가동 냉각탑 ID 리스트
    * `active_tower_count`: 가동 냉각탑 대수
    * `total_fan_power_watts`: 총 팬 소비 전력 (Watts)
    * `total_active_capacity_watts`: 가동 냉각탑의 총 방열 용량 (Watts)

## 3. 주요 모듈 및 클래스 상세

### 3.1. `EnthalpyCalculator` 클래스

* **목적**: 공기의 온도와 습도로부터 엔탈피를 계산하거나, 엔탈피와 초기 온도로부터 새로운 온도와 습도를 추정합니다. (현재 구현은 매우 단순화된 Placeholder임)
* **주요 메소드**:
    * `calculate_enthalpy(temperature_c, relative_humidity_percent, pressure_pa=101325)`
        * **입력**: `temperature_c` (float, 섭씨 온도), `relative_humidity_percent` (float, 상대 습도 %), `pressure_pa` (float, 대기압 Pa, 기본값 101325)
        * **출력**: `float`, 계산된 엔탈피 (J/kg 건공기)
    * `calculate_temp_humidity_from_enthalpy(enthalpy_j_kg, initial_temp_c, pressure_pa=101325)`
        * **입력**: `enthalpy_j_kg` (float, 엔탈피 J/kg), `initial_temp_c` (float, 초기 온도 섭씨), `pressure_pa` (float, 대기압 Pa)
        * **출력**: `tuple (float, float)`, (새로운 온도 섭씨, 새로운 상대 습도 %)

### 3.2. `generate_dummy_data` 함수

* **목적**: 모델 학습 및 시스템 테스트를 위한 상세 시계열 더미 데이터를 생성합니다.
* **파라미터**:
    * `num_rows` (int): 생성할 데이터 행(시간 스텝) 수 (기본값 1000)
    * `num_oac` (int): 생성할 OAC 유닛 수 (기본값 14)
    * `num_low_ch` (int): 생성할 저온 냉동기 수 (기본값 8)
    * `num_high_ch` (int): 생성할 고온 냉동기 수 (기본값 8)
    * `num_ct` (int): 생성할 냉각탑 수 (기본값 9)
* **출력**: `pandas.DataFrame`. 다음을 포함하는 매우 넓은 테이블:
    * `datetime`: 시간 정보
    * `outdoor_temp_c`, `outdoor_rh_percent`: 외기 온도 및 습도
    * OAC 관련 (각 `oac_i` 유닛별, 각 코일별): `set_point_temp_c` (또는 `set_point_rate`), `coil_open_rate`, `후단온도_c`, `후단습도_percent`, `토출온도_c`, `토출노점온도_c`, `토출압력_pa`
    * 시스템 전체 부하: `total_precooling_load_watts`, `total_cooling_load_watts`
    * 냉동기 시스템 레벨 (저온/고온별): `main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_percent`, `main_supply_압력_pa`, `main_supply_온도_c`, `main_return_압력_pa`, `main_return_온도_c`, `active_count` (총 가동 대수)
    * 개별 냉동기 (저온/고온별, 각 `chiller_j` 유닛별): `supply_냉수온도_c`, `return_냉수온도_c`, `supply_냉각수온도_c`, `return_냉각수온도_c`
    * 개별 냉각탑 (각 `tower_k` 유닛별): `supply_냉각수온도_c`, `return_냉각수온도_c`, `supply_수조레벨_percent`

### 3.3. `OACModel` 클래스

* **목적**: OAC(공기조화기)의 각 코일(예열, 예냉, 냉각, 재열, 가습)의 개도율을 예측하는 모델을 학습하고, 학습된 모델 또는 규칙 기반으로 개도율 및 공기 상태 변화를 예측합니다. 현재는 대표 OAC(0번)의 데이터를 사용하여 코일 타입별 단일 모델을 학습합니다.
* **주요 속성**:
    * `trained_models` (dict): 학습된 코일별 모델 저장 (`{coil_type: RandomForestRegressor_model}`)
    * `feature_names_per_coil` (dict): 코일별 모델 학습 시 사용된 일반화된 피처 이름 리스트 저장
    * `design_delta_h` (dict): 코일별 설계 엔탈피 변화량 (개도율 100% 기준)
* **주요 메소드**:
    * `train_coil_models(df_train_data_full)`
        * **입력**: `df_train_data_full` (pandas.DataFrame, `generate_dummy_data`로부터 생성된 전체 더미 데이터)
        * **동작**: 대표 OAC(oac_0)의 데이터를 사용하여 각 코일 타입(`preheating`, `precooling` 등)별 개도율 예측 모델(`RandomForestRegressor`)을 학습하고 `self.trained_models` 및 `self.feature_names_per_coil`에 저장합니다.
        * **출력**: 없음 (내부 모델 학습)
    * `predict_coil_open_rates(outdoor_temp_c, outdoor_rh_percent, oac_set_points, modified_coil_states=None)`
        * **입력**: 시스템 입력과 동일 (`outdoor_temp_c`, `outdoor_rh_percent`, `oac_set_points`, `modified_coil_states`)
        * **동작**: 학습된 모델이 있으면 모델을 사용하여, 없으면 규칙 기반(`_predict_coil_open_rates_rule_based`)으로 각 코일의 개도율을 순차적으로 예측하고, 각 코일 통과 후의 공기 상태(온도, 습도, 엔탈피)를 계산합니다.
        * **출력**: `dict`, 시스템 출력의 `oac_predictions_per_unit` 부분과 동일한 구조.
    * `_predict_coil_open_rates_rule_based(...)`
        * **동작**: 모델이 없을 경우 사용되는 규칙 기반 예측 로직.

### 3.4. `ChillerModel` 클래스

* **목적**: 냉동기 시스템(저온/고온 타입별)의 주요 운전 지표(메인 차압, 메인 차압 밸브 개도율, 서브 차압 밸브 개도율)를 예측하는 모델을 학습하고, 이를 이용해 예측값을 제공합니다.
* **주요 속성**:
    * `trained_models` (dict): 학습된 냉동기 타입별/타겟별 모델 저장 (`{chiller_type: {target_name: RandomForestRegressor_model}}`)
    * `feature_names_chiller` (dict): 냉동기 타입별 모델 학습 시 사용된 피처 이름 리스트 저장
* **주요 메소드**:
    * `train_pressure_valve_models(df_train_data)`
        * **입력**: `df_train_data` (pandas.DataFrame, 전체 더미 데이터)
        * **동작**: 저온/고온 냉동기 타입별로, 시스템 전체 부하와 해당 타입의 총 가동 대수를 입력 특징으로 하여, `main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_percent`를 예측하는 모델(`RandomForestRegressor`)들을 학습하고 `self.trained_models` 및 `self.feature_names_chiller`에 저장합니다.
        * **출력**: 없음 (내부 모델 학습)
    * `predict_pressure_valve(current_total_load_watts, num_active_chillers_of_type, chiller_type="low_temp")`
        * **입력**: `current_total_load_watts` (float, 해당 타입 냉동기의 현재 총 부하량 Watts), `num_active_chillers_of_type` (int, 해당 타입 냉동기의 현재 가동 대수), `chiller_type` (str, "low_temp" 또는 "high_temp")
        * **동작**: 학습된 모델을 사용하여 해당 냉동기 타입의 시스템 레벨 차압/밸브 상태를 예측합니다.
        * **출력**: `dict`, 예: `{"main_차압압력_bar": 1.45, "main_차압개도율_percent": 62.0, "sub_차압개도율_percent": 55.0}`

### 3.5. `ChillerOptimizer` 클래스

* **목적**: 주어진 냉각 부하에 대해 저온 또는 고온 냉동기들의 최적 가동 대수 및 각 냉동기별 부하 분담률을 결정합니다. 최적화 목표는 평균 부하율 55% 근접 및 총 전력 소비 최소화입니다.
* **주요 속성**:
    * `chiller_specs` (dict): 냉동기 타입별(low_temp, high_temp) 각 냉동기 스펙 (ID, 최대 용량, COP 곡선)
    * `target_load_percentage` (float): 목표 평균 부하율 (예: 0.55)
* **주요 메소드**:
    * `_get_cop_at_load(cop_points, load_percentage)`: 주어진 부하율에서 COP 값을 (선형 보간하여) 계산합니다. (내부 사용)
    * `optimize_chiller_operation(required_total_load_watts, chiller_type="low_temp")`
        * **입력**: `required_total_load_watts` (float, 필요한 총 냉각 부하 Watts), `chiller_type` (str, "low_temp" 또는 "high_temp")
        * **동작**: 1대부터 최대 가동 대수까지 모든 경우를 탐색하여, 각 냉동기의 부하율 제약(min/max)을 만족하면서 목표 평균 부하율에 가장 가깝고 총 전력 소비가 적은 조합을 찾습니다. COP가 높은 냉동기를 우선적으로 고려합니다.
        * **출력**: `tuple (optimal_active_count, active_chiller_ids, load_percentages, total_power, avg_load_percentage, meets_target_load_rate)`

### 3.6. `CoolingTowerOptimizer` 클래스

* **목적**: 냉동기에서 발생하는 총 방열 부하를 처리하기 위한 냉각탑의 최적 가동 조합(가동 대수 및 ID)을 결정합니다. 최적화 목표는 총 방열 부하를 만족시키면서 냉각탑 팬의 총 전력 소비를 최소화하는 것입니다.
* **주요 속성**:
    * `tower_specs` (list): 각 냉각탑의 스펙 (ID, 최대 방열 용량, 팬 소비 전력). 용량이 상이할 수 있음을 고려.
    * `sorted_towers` (list): 용량 대비 팬 전력 효율이 좋은 순으로 정렬된 냉각탑 스펙 리스트.
* **주요 메소드**:
    * `optimize_tower_operation(total_heat_rejection_load_watts)`
        * **입력**: `total_heat_rejection_load_watts` (float, 처리해야 할 총 방열 부하 Watts)
        * **동작**: 효율이 좋은 냉각탑부터 순서대로 가동(Greedy 접근)하여 필요 용량을 만족하는 최소한의 냉각탑 조합을 찾습니다.
        * **출력**: `tuple (active_tower_ids, total_fan_power_watts, total_active_capacity_watts, active_tower_count)`

### 3.7. `HVACSystemFacade` 클래스

* **목적**: 시스템의 모든 하위 모듈(OAC 모델, 냉동기 모델/최적화, 냉각탑 최적화 등)을 통합하고 전체 예측 및 최적화 프로세스를 관리하는 인터페이스 역할을 합니다.
* **주요 속성**:
    * `oac_model` (`OACModel` 인스턴스)
    * `chiller_model` (`ChillerModel` 인스턴스)
    * `chiller_optimizer` (`ChillerOptimizer` 인스턴스)
    * `cooling_tower_optimizer` (`CoolingTowerOptimizer` 인스턴스)
* **주요 메소드**:
    * `__init__(oac_model_instance, chiller_model_instance)`: 학습된 OAC 및 냉동기 모델 인스턴스를 주입받아 초기화합니다.
    * `predict_and_optimize(outdoor_temp_c, outdoor_rh_percent, oac_set_points, chilled_water_supply_set_temp, modified_oac_coil_states=None)`
        * **입력**: "2.1. 주요 입력 (Facade 기준)" 참조.
        * **동작**:
            1.  `OACModel`을 사용하여 대표 OAC의 코일별 개도율 및 공기 상태 변화 예측.
            2.  예측된 OAC 코일 부하(예냉, 냉각)를 기반으로 전체 시스템의 총 예냉 부하 및 총 냉각 부하 계산.
            3.  `ChillerOptimizer`를 사용하여 저온 및 고온 냉동기의 최적 가동 대수, 부하 분배, 예상 소비 전력 계산.
            4.  `ChillerModel`을 사용하여 최적화된 운전 조건에서의 저온/고온 냉동기 시스템 차압/밸브 상태 예측.
            5.  총 냉동기 부하와 소비 전력을 합산하여 총 시스템 방열 부하 계산.
            6.  `CoolingTowerOptimizer`를 사용하여 냉각탑 최적 가동 조합 결정.
        * **출력**: "2.2. 주요 출력 (Facade 기준)" 참조.

## 4. 메인 실행 블록 (`if __name__ == '__main__':`)

* **목적**: 전체 시스템의 작동 흐름을 시연합니다.
* **주요 단계**:
    1.  `generate_dummy_data` 함수를 호출하여 상세 더미 학습 데이터 생성.
    2.  `OACModel` 및 `ChillerModel` 객체를 생성하고, 생성된 더미 데이터로 각 모델의 `train_` 메소드를 호출하여 내부 모델들을 학습.
    3.  학습된 모델 인스턴스들을 `HVACSystemFacade` 생성자에 주입하여 Facade 객체 생성.
    4.  미리 정의된 하절기 외기 조건 및 OAC 설정값을 사용하여 `hvac_system.predict_and_optimize` 메소드를 호출.
    5.  반환된 최종 예측 및 최적화 결과를 콘솔에 주요 항목 위주로 출력.

