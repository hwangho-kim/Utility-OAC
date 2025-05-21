```text
# HVAC 최적화 시스템 명세서 (V12 수정본)

## 1\. 시스템 개요

본 문서는 반도체 클린룸의 공조기(HVAC) 시스템에 대한 부하 예측 및 관리 최적화 머신러닝 알고리즘의 명세를 기술합니다. 이 시스템은 외기 조건, OAC(Out Air Controller) 유닛별 코일 목표 온도, 그리고 필요에 따라 사용자가 수동으로 지정한 코일 개도율을 입력받습니다. 이를 바탕으로 각 OAC 유닛의 코일별 개도율 및 공기 상태 변화를 예측하고, 이 예측을 기반으로 저온/고온 냉동기 시스템 및 냉각탑 시스템의 최적 운전 상태(가동 대수, 부하율, 예상 소비 전력 등)와 시스템 레벨의 주요 지표(차압, 밸브 개도율 등)를 도출합니다.

시스템은 다음과 같은 주요 기능을 포함합니다:

  * **데이터 처리**:
      * 상세 더미 데이터 생성 (기본 10분 주기). OAC는 층(F3/F4) 및 AI 모드 유무 정보가 이름에 포함된 피처를 가집니다. 냉동기는 개별 전력 피처를 포함하며, 냉각탑은 수조 레벨 피처를 포함합니다.
      * 외부 CSV 파일(예: 초당 데이터) 로드 시 10분 단위 리샘플링 및 전처리 기능.
      * 데이터 소스(더미 또는 CSV) 선택 기능.
      * 냉동기 및 냉각탑의 가동 상태(가동 대수 등)를 원본 데이터셋의 피처(개별 냉동기 전력, 냉각탑 수조 레벨)로부터 동적으로 파생.
  * **모델 학습 및 관리**:
      * 14대의 OAC 유닛 각각에 대해, 5개 코일(예열, 예냉, 냉각, 재열, 가습)별로 독립적인 개도율 예측 모델(RandomForestRegressor) 학습.
      * 저온/고온 냉동기 시스템별로 주요 지표(메인 차압, 메인 차압 개도율, **층별 서브 차압 개도율(F3/F4)**) 예측 모델(RandomForestRegressor) 학습. 학습 시 냉동기 메인 냉수 온도차(delta\_T)와 파생된 가동 대수를 주요 특징으로 사용.
      * **학습된 OAC 및 냉동기 모델을 파일로 저장하고, 필요시 불러와 사용하는 기능 (`joblib` 라이브러리 활용).**
  * **예측 및 최적화**:
      * 학습된 모델을 사용하여 각 OAC 유닛의 코일별 개도율 및 공기 상태 변화 예측 (**엔탈피 단위 kJ/kg 사용**). OAC 유닛별 개별 목표 온도 설정(Set Point) 적용.
      * OAC 유닛의 AI 모드 유무에 따라 예냉/냉각 부하를 저온 또는 고온 냉동기에 적절히 할당하여 시스템 전체의 냉각/예냉 부하 계산.
      * 계산된 부하를 바탕으로 냉동기(저온/고온) 최적 운전 조합 결정 (**평균 부하율 50% 내외 권장, 최대 부하율 70% 제한**).
      * 최적화된 냉동기 운전 조건에 기반하여 **메인 냉수 온도차(delta\_T)를 추정**하고, 이를 사용하여 냉동기 시스템 지표 예측.
      * 냉동기 운전에 따른 총 방열 부하를 계산하고, 냉각탑 최적 운전 조합 결정.
  * **상세 결과 출력**:
      * OAC 유닛별 상세 예측 결과(목표 SP, 입/출구 상태, 엔탈피 변화 등) 출력.
      * OAC 개도율 요약 테이블(층, AI 모드 정보 포함) 출력.
      * 학습된 모델 정보(타입, 주요 파라미터, 피처 중요도 등) 확인 기능 예시 제공.
      * 최종 최적화 결과(냉동기, 냉각탑) 출력.
      * `DEBUG_MODE` 플래그를 통해 Facade 내부의 중간 계산 과정 상세 출력 제어.

## 2\. 시스템 전체 입력 및 출력 (`HVACSystemFacade` 기준)

### 2.1. 주요 입력 (`predict_and_optimize` 메소드)

  * `outdoor_temp_c` (float): 현재 외기 온도 (섭씨).
  * `outdoor_rh_percent` (float): 현재 외기 상대 습도 (%).
  * `oac_set_points_by_unit` (dict): **OAC 유닛별, 각 코일 단계별 목표 온도 설정값.**
      * 외부 키: OAC 유닛 이름 문자열 (예: `'F03_OAC_00_nonAI'`). 이 이름은 `get_oac_name_prefix` 함수를 통해 생성된 규칙을 따릅니다.
      * 내부 키: 코일 종류 문자열 (`'preheating'`, `'precooling'`, `'cooling'`, `'reheating'`, `'humidification'`).
      * 내부 값: 해당 코일의 목표 출구 온도 (섭씨).
      * 예시: `{'F03_OAC_00_nonAI': {'preheating': 18.0, 'cooling': 12.0, ...}, 'F04_OAC_00_AI': {'preheating': 18.2, 'cooling': 11.8, ...}}`
  * `oac_meta_data_by_unit` (dict): **OAC 유닛별 메타데이터 (층, AI 모드 정보).**
      * 외부 키: OAC 유닛의 일반 ID 문자열 (예: `'oac_0'`, `'oac_1'`, ...). 이 키는 `HVACSystemFacade` 내부에서 `oac_predictions_all_units`의 키(예: `F03_OAC_00_nonAI`)와 매칭됩니다.
      * 내부 키: `'floor'` (str, 예: 'F3'), `'ai_mode'` (bool, 예: True).
      * 예시: `{'oac_0': {'floor': 'F3', 'ai_mode': False}, 'oac_7': {'floor': 'F4', 'ai_mode': True}}`
  * `chilled_water_supply_set_temp` (float): 냉수 공급 설정 온도 (섭씨, 현재 로직에서는 직접적인 제어 변수보다는 참고용).
  * `modified_oac_coil_states_all_units` (dict, Optional): 사용자가 수동으로 변경한 특정 OAC 코일의 개도율. OAC 유닛별, 코일별로 지정 가능.
      * 외부 키: OAC 유닛 이름 문자열 (예: `'F03_OAC_00_nonAI'`).
      * 내부 키: 코일 종류 문자열.
      * 내부 값: 해당 코일의 강제 설정 개도율 (0.0 \~ 1.0).

### 2.2. 주요 출력 (`predict_and_optimize` 메소드)

딕셔너리 형태로 반환되며, 주요 키는 다음과 같습니다:

  * `oac_predictions_all_units` (dict): 시스템 내 모든 OAC 유닛 각각에 대한 코일별 예측 결과.
      * 키: OAC 유닛 이름 문자열 (예: `'F03_OAC_00_nonAI'`).
      * 값 (각 OAC 유닛별 딕셔너리):
          * 키: 코일 종류 문자열.
          * 값 (각 코일별 딕셔너리): `open_rate`, `inlet_temp_c`, `inlet_rh_percent`, `inlet_h_j_kg` (kJ/kg), `outlet_temp_c`, `outlet_rh_percent`, `outlet_h_j_kg` (kJ/kg), `delta_h_j_kg` (kJ/kg), `target_set_point_temp_c`.
          * `final_outlet_temp_c`, `final_outlet_rh_percent`, `final_outlet_h_j_kg` (kJ/kg).
  * `calculated_loads` (dict): 계산된 총 시스템 부하.
      * `total_low_temp_chiller_load_watts` (float): 모든 OAC에서 저온 냉동기가 담당해야 하는 총 부하량 (Watts).
      * `total_high_temp_chiller_load_watts` (float): 모든 OAC에서 고온 냉동기가 담당해야 하는 총 부하량 (Watts).
  * `low_temp_chiller_optimization` (dict): 저온 냉동기 시스템 최적화 및 예측 결과.
      * `optimal_active_count`, `active_chiller_ids`, `load_distribution_percentage_per_active_chiller`, `total_estimated_power_watts`, `average_load_percentage_of_active_chillers`, `meets_target_load_rate`.
      * `predictions` (dict): `main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_F3_percent`, `sub_차압개도율_F4_percent`.
  * `high_temp_chiller_optimization` (dict): 고온 냉동기 시스템 최적화 및 예측 결과 (저온 냉동기와 동일 구조).
  * `cooling_tower_optimization` (dict): 냉각탑 최적화 결과. `total_heat_rejection_load_watts`, `active_tower_ids`, `active_tower_count`, `total_fan_power_watts`, `total_active_capacity_watts`.

## 3\. 주요 모듈 및 클래스 상세

### 3.1. `EnthalpyCalculator` 클래스

  * **목적**: 공기의 온도와 상대습도로부터 엔탈피를 계산하고, 엔탈피와 초기 온도로부터 새로운 온도와 상대습도를 추정합니다. 모든 엔탈피 계산은 **kJ/kg** 단위를 사용합니다.
  * **주요 메소드**:
      * `calculate_enthalpy(temperature_c, relative_humidity_percent, pressure_pa=101325)`: 엔탈피(kJ/kg) 계산.
      * `calculate_temp_humidity_from_enthalpy(enthalpy_kj_kg, initial_temp_c, pressure_pa=101325)`: 엔탈피(kJ/kg)로부터 온도/습도 추정.

### 3.2. `get_oac_name_prefix` 함수

  * **목적**: OAC의 전체 시스템 내 순차 인덱스(0\~13)를 입력받아, 층(F3/F4), 해당 층 내 인덱스(00\~06), AI 모드 유무(AI/nonAI) 정보를 포함하는 표준화된 이름 접두사(예: "F03\_OAC\_00\_nonAI\_")를 생성합니다.
  * **파라미터**:
      * `oac_global_idx` (int): 전체 OAC 시스템에서의 0부터 시작하는 인덱스.
      * `num_oac_per_floor` (int, 기본값 7): 한 층당 OAC 수.
  * **출력**: `str`, 생성된 OAC 이름 접두사.

### 3.3. `generate_dummy_data` 함수

  * **목적**: 모델 학습 및 시스템 테스트를 위한 상세 시계열 더미 데이터를 지정된 주기(기본 10분)로 생성합니다. **사용자가 명시한 피처 목록에 해당하는 데이터만 생성**하며, 파생 피처(예: 엔탈피, 냉동기 가동 대수 등)는 이 함수에서 직접 생성하여 데이터프레임에 포함시키지 않습니다. OAC 피처 이름은 `get_oac_name_prefix`를 사용하여 표준화된 명명 규칙을 따릅니다.
  * **파라미터**: `num_days`, `freq_minutes`, `num_oac`, `num_low_ch`, `num_high_ch`, `num_ct`.
  * **출력**: `pandas.DataFrame`. 명시된 스키마에 따른 컬럼들만 포함.
      * **OAC**: 각 OAC 유닛(이름에 층/AI모드 포함)에 대해 `_set_point_temp_c` (모든 코일 목표 온도), `_coil_open_rate`, `_후단온도_c`, `_후단습도_percent` (가습 코일 제외), `_토출온도_c`, `_토출노점온도_c`, `_토출압력_pa`.
      * **냉동기**: 각 타입(저온/고온)에 대해 `_main_차압압력_bar`, `_main_차압개도율_percent`, `_sub_차압개도율_F3_percent`, `_sub_차압개도율_F4_percent`, `_main_supply_압력_pa`, `_main_supply_온도_c`, `_main_return_압력_pa`, `_main_return_온도_c`. 그리고 각 개별 냉동기 유닛에 대해 `_supply_냉수온도_c`, `_return_냉수온도_c`, `_supply_냉각수온도_c`, `_return_냉각수온도_c`, `_전력_kW`.
      * **냉각탑**: 각 개별 냉각탑 유닛에 대해 `_supply_냉각수온도_c`, `_return_냉각수온도_c`, `_supply_수조레벨_percent`.
      * **외기**: `outdoor_temp_c`, `outdoor_rh_percent`.

### 3.4. `create_dummy_csv_second_data` 함수

  * **목적**: CSV 파일 처리 기능 테스트를 위한 초당 더미 CSV 파일을 생성합니다. `generate_dummy_data`와 유사하게 명시된 피처들만 포함합니다. OAC 이름 규칙을 따릅니다.
  * **파라미터**: `filepath`, `num_hours`, `num_oac`, `num_low_ch`, `num_high_ch`, `num_ct`.
  * **출력**: 지정된 경로에 CSV 파일 생성.

### 3.5. `derive_chiller_active_counts` 함수

  * **목적**: 입력 데이터프레임에서 각 냉동기 타입별(저온/고온) 개별 냉동기의 **`_전력_kW`** 피처를 기반으로 현재 가동 중인 냉동기 대수를 계산하여 `f'{temp_type_prefix}_active_count'` 컬럼을 생성하고 데이터프레임에 추가합니다.
  * **파라미터**:
      * `df` (pandas.DataFrame): 입력 데이터프레임.
      * `num_low_ch` (int, 기본값 8), `num_high_ch` (int, 기본값 8): 저온/고온 냉동기 수.
      * `power_on_threshold_kw` (float, 기본값 0.1): 냉동기가 가동 중이라고 판단하는 최소 전력 임계값 (kW).
  * **출력**: `pandas.DataFrame`, `_active_count` 컬럼이 추가된 데이터프레임.

### 3.6. `derive_cooling_tower_active_details` 함수

  * **목적**: 입력 데이터프레임에서 각 냉각탑의 `_supply_수조레벨_percent`를 기반으로 가동 상태(`_is_active`)를 판단하고, 총 가동 냉각탑 수(`derived_total_active_cooling_towers`)를 계산하여 컬럼으로 추가합니다.
  * **파라미터**:
      * `df` (pandas.DataFrame): 입력 데이터프레임.
      * `num_ct` (int, 기본값 9): 냉각탑 수.
      * `tank_level_off_threshold` (float, 기본값 1.0): 이 레벨 미만이면 비가동으로 판단하는 임계값 (%).
  * **출력**: `pandas.DataFrame`, `_is_active` 및 `derived_total_active_cooling_towers` 컬럼이 추가된 데이터프레임.

### 3.7. `load_and_preprocess_data` 함수

  * **목적**: 지정된 데이터 소스('dummy' 또는 'csv')에 따라 데이터를 로드하고, CSV인 경우 전처리(10분 단위 리샘플링)를 수행한 후, `derive_chiller_active_counts` 및 `derive_cooling_tower_active_details` 함수를 호출하여 필요한 파생 피처를 추가합니다.
  * **파라미터**: `data_source`, `csv_filepath`, `num_days_dummy`, `freq_minutes_dummy`.
  * **출력**: `pandas.DataFrame`. 모델 학습 및 예측에 사용될, 파생 피처가 포함된 데이터.

### 3.8. `OACModel` 클래스

  * **목적**: 시스템 내 모든 OAC 유닛 각각에 대해, 각 코일의 개도율을 예측하는 독립적인 모델들을 학습, 예측, 저장 및 로드합니다. 각 OAC 유닛은 개별적인 목표 온도 설정값을 입력받아 사용하며, 가습 코일 후의 상태는 OAC의 최종 토출 상태입니다. 엔탈피 관련 계산은 kJ/kg 단위를 사용합니다. OAC 피처 이름은 `get_oac_name_prefix`로 생성된 표준화된 이름을 따릅니다.
  * **주요 속성**: `num_oac_units_in_system`, `trained_models`, `feature_names_per_coil`, `design_delta_h` (kJ/kg 단위).
  * **주요 메소드**:
      * `save_models(directory_path)` / `load_models(directory_path)`: 모델 저장/로드.
      * `train_coil_models(df_train_data_full)`: OAC 유닛별, 코일별 모델 학습. 입력 `df_train_data_full`은 명시된 피처만 포함. 엔탈피 등 필요한 파생 피처는 메소드 내부에서 동적으로 계산.
      * `predict_coil_open_rates(outdoor_temp_c, outdoor_rh_percent, oac_set_points_by_unit, modified_oac_coil_states_all_units=None)`: OAC 유닛별 예측 수행. `oac_set_points_by_unit`으로 유닛별 SP 입력받음.
      * `_predict_single_oac_rule_based(...)`: 단일 OAC 규칙 기반 예측.

### 3.9. `ChillerModel` 클래스

  * **목적**: 냉동기 시스템 레벨 지표 예측 모델 학습, 예측, 저장 및 로드.
  * **주요 속성**: `trained_models`, `feature_names_chiller`.
  * **주요 메소드**:
      * `save_models(directory_path)` / `load_models(directory_path)`: 모델 저장/로드.
      * `train_pressure_valve_models(df_train_data)`: 모델 학습. 입력 `df_train_data`는 `_active_count`가 파생되어 포함된 상태. `_main_delta_t_c`는 이 메소드 내에서 `main_supply/return_온도_c`로부터 동적 계산.
          * **학습 특징**: 동적으로 계산된 `_main_delta_t_c`와 파생된 `_active_count`.
          * **예측 대상**: `main_차압압력_bar`, `main_차압개도율_percent`, `sub_차압개도율_F3_percent`, `sub_차압개도율_F4_percent`.
      * `predict_pressure_valve(current_main_delta_t_c, num_active_chillers_of_type, chiller_type="low_temp")`: 예측값 반환.

### 3.10. `ChillerOptimizer` 클래스

  * **목적**: 주어진 냉각 부하에 대해 냉동기 최적 가동 대수 및 부하 분담률 결정 (평균 부하율 50% 내외, 최대 부하율 70% 제약).
  * **주요 속성**: `chiller_specs`, `target_load_percentage` (0.50), `min_load_percentage` (0.30), `max_load_percentage` (0.70).
  * **주요 메소드**:
      * `_get_cop_at_load(...)`: COP 계산.
      * `optimize_chiller_operation(...)`: 최적 운전 조합 탐색.

### 3.11. `CoolingTowerOptimizer` 클래스

  * **목적**: 냉동기에서 발생하는 총 방열 부하를 처리하기 위한 냉각탑의 최적 가동 조합을 결정합니다.
  * **주요 속성**: `tower_specs`, `sorted_towers`.
  * **주요 메소드**:
      * `optimize_tower_operation(total_heat_rejection_load_watts)`: 최적 가동 조합 탐색.

### 3.12. `HVACSystemFacade` 클래스

  * **목적**: 시스템의 모든 하위 모듈을 통합하고 전체 예측 및 최적화 프로세스를 관리하는 인터페이스.
  * **주요 속성**: `oac_model`, `chiller_model`, `chiller_optimizer`, `cooling_tower_optimizer`.
  * **주요 메소드**:
      * `__init__(oac_model_instance, chiller_model_instance)`.
      * `_estimate_chiller_main_delta_t(...)`: 냉동기 메인 냉수 온도차 추정.
      * `predict_and_optimize(outdoor_temp_c, outdoor_rh_percent, oac_set_points_by_unit, oac_meta_data_by_unit, ...)`:
          * **입력**: `oac_set_points_by_unit` (유닛별 SP), `oac_meta_data_by_unit` (유닛별 층/AI모드 정보).
          * **동작**:
            1.  `OACModel` 호출 시 `oac_set_points_by_unit` 전달하여 OAC 예측.
            2.  `oac_meta_data_by_unit`의 AI 모드 정보를 참조하여 저온/고온 냉동기 부하 분리 계산.
            3.  (이하 V6 명세와 유사)

## 4\. 메인 실행 블록 (`if \_\_name\_\_ == '\_\_main\_\_':`)

  * **목적**: 전체 시스템의 작동 흐름 시연.
  * **주요 단계**:
    1.  전역 설정 (`DEBUG_MODE`, `TRAIN_NEW_MODELS`, `MODEL_SAVE_DIR`, 임계값들).
    2.  데이터 소스 선택 및 `load_and_preprocess_data` 호출 (내부적으로 `active_count` 등 파생).
    3.  `TRAIN_NEW_MODELS` 플래그에 따라 모델 신규 학습/저장 또는 기존 모델 로드.
    4.  학습된 모델 정보(일부) 출력.
    5.  `HVACSystemFacade` 객체 생성.
    6.  하절기 시나리오 실행: OAC 유닛별 SP 및 메타데이터(`oac_meta_data_by_unit_summer`) 생성하여 Facade에 전달.
    7.  결과 출력 (OAC 유닛별 상세 정보, 개도율 요약 테이블, 냉동기/냉각탑 최적화 결과).

<!-- end list -->
