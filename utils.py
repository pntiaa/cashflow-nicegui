import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from nicegui import ui, app

# --- State Management ---
# For NiceGUI, we can use app.storage.user for per-user state.
# Since this is likely a local app, we'll initialize it in a way that's easy to access.

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
DEFAULTS_FILE = DATA_DIR / "defaults.json"

def load_defaults():
    if DEFAULTS_FILE.exists():
        with open(DEFAULTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_defaults(defaults):
    with open(DEFAULTS_FILE, "w") as f:
        json.dump(defaults, f, indent=4)

def ensure_state_init():
    """Initializes the app.storage.user with default values if not present."""
    # Note: app.storage.user is only available within a request context or after app.run()
    # We will call this at the beginning of each page or in a shared layout.
    
    s = app.storage.user
    if "current_project" not in s:
        s["current_project"] = None
    if "production_cases" not in s:
        s["production_cases"] = {}
    if "development_cases" not in s:
        s["development_cases"] = {}
    if "price_cases" not in s:
        s["price_cases"] = {}
    if "cashflow_results" not in s:
        s["cashflow_results"] = {}
    
    # UI transient states
    if "profile" not in s:
        s["profile"] = None
    if "tc_data" not in s:
        s["tc_data"] = None
    if "prod_data" not in s:
        s["prod_data"] = None
    if "drilling_plan_results" not in s:
        s["drilling_plan_results"] = None
    
    if "defaults" not in s:
        s["defaults"] = load_defaults()
    
    # Initialize input parameters from defaults if not present
    if s["defaults"]:
        d = s["defaults"]
        p_keys = {
            "qi_input": d["production"]["qi_mmcfd"],
            "well_eur_input": d["production"]["well_eur_bcf"],
            "prod_dur_input": d["production"]["prod_duration"],
            "giip_input": d["production"]["giip_bcf"],
            "oiip_input": d["production"]["oiip_mmbbl"],
            "drilling_rate_input": d["production"]["drilling_rate"],
            "max_rate_input": d["production"]["max_prod_rate"],
            "sunk_cost_input": d["development"]["sunk_cost"],
            "exp_start_year_input": d["development"]["exp_start_year"],
            "dev_start_year_input": d["development"]["dev_start_year"],
            "drill_start_year_input": d["development"]["drill_start_year"],
            "dev_case_input": d["development"]["dev_case"],
            "feas_study_input": d["development"]["study_costs"]["feasibility"]["cost"],
            "feas_study_t": d["development"]["study_costs"]["feasibility"]["timing"],
            "feas_study_d": d["development"]["study_costs"]["feasibility"]["duration"],
            "concept_study_input": d["development"]["study_costs"]["concept"]["cost"],
            "concept_study_t": d["development"]["study_costs"]["concept"]["timing"],
            "concept_study_d": d["development"]["study_costs"]["concept"]["duration"],
            "feed_cost_input": d["development"]["study_costs"]["feed_fpso"]["cost"],
            "feed_cost_t": d["development"]["study_costs"]["feed_fpso"]["timing"],
            "feed_cost_d": d["development"]["study_costs"]["feed_fpso"]["duration"],
            "pm_others_input": d["development"]["study_costs"]["pm_others_per_well"],
            "drilling_cost_input": d["development"]["facility_capex"]["drilling_per_well"],
            "subsea_cost_input": d["development"]["facility_capex"]["subsea_per_well"],
            "fpso_cost_input": d["development"]["facility_capex"]["fpso_fpso"]["cost"],
            "fpso_cost_t": d["development"]["facility_capex"]["fpso_fpso"]["timing"],
            "fpso_cost_d": d["development"]["facility_capex"]["fpso_fpso"]["duration"],
            "pipeline_cost_input": d["development"]["facility_capex"]["pipeline_fpso"]["cost"],
            "pipeline_cost_t": d["development"]["facility_capex"]["pipeline_fpso"]["timing"],
            "pipeline_cost_d": d["development"]["facility_capex"]["pipeline_fpso"]["duration"],
            "terminal_cost_input": d["development"]["facility_capex"]["terminal_fpso"]["cost"],
            "terminal_cost_t": d["development"]["facility_capex"]["terminal_fpso"]["timing"],
            "terminal_cost_d": d["development"]["facility_capex"]["terminal_fpso"]["duration"],
            "opex_per_bcf_input": d["development"]["opex_abex"]["opex_per_bcf"],
            "opex_fixed_input": d["development"]["opex_abex"]["opex_fixed"],
            "abex_per_well_input": d["development"]["opex_abex"]["abex_per_well"],
            "abex_fpso_input": d["development"]["opex_abex"]["abex_fpso_fpso"],
            "base_year_input": d["economics"]["base_year"],
            "discount_rate_input": d["economics"]["discount_rate"],
            "exchange_rate_input": d["economics"]["exchange_rate"],
            "cost_inflation_input": d["economics"]["cost_inflation"],
            "useful_life_input": d["economics"]["useful_life"],
            "depreciation_method_input": d["economics"]["depreciation_method"],
            "oil_init_input": d["price_deck"]["oil_init"],
            "gas_init_input": d["price_deck"]["gas_init"],
            "price_start_year_input": d["price_deck"]["start_year"],
            "price_end_year_input": d["price_deck"]["end_year"],
            "price_inflation_input": d["price_deck"]["inflation_pct"],
            "price_cap_2x_input": d["price_deck"]["cap_2x"],
            "q_dev_start_year": d["development"]["dev_start_year"],
            "q_sunk_cost": d["development"]["sunk_cost"],
            "q_exp_start_year": d["development"]["exp_start_year"],
        }
        for k, v in p_keys.items():
            if k not in s:
                s[k] = v
        
        # Also initialize price_deck_oil and gas if not present
        if "price_deck_oil" not in s:
            start = d["price_deck"]["start_year"]
            end = d["price_deck"]["end_year"]
            init_oil = d["price_deck"]["oil_init"]
            init_gas = d["price_deck"]["gas_init"]
            s["price_deck_oil"] = {y: init_oil for y in range(start, end + 1)}
            s["price_deck_gas"] = {y: init_gas for y in range(start, end + 1)}

def sanitize_data(data):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if isinstance(k, (np.integer, np.int64)):
                k = int(k)
            elif isinstance(k, (np.floating, np.float64)):
                k = float(k)
            new_dict[k] = sanitize_data(v)
        return new_dict
    elif isinstance(data, list):
        return [sanitize_data(i) for i in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return sanitize_data(data.tolist())
    else:
        return data

def save_project(project_name: str):
    if not project_name:
        return
    file_path = DATA_DIR / f"{project_name}.json"
    
    s = app.storage.user
    raw_data = {
        "production_cases": s.get("production_cases", {}),
        "development_cases": serialize_dev_cases(s.get("development_cases", {})),
        "price_cases": s.get("price_cases", {}),
        "cashflow_results": serialize_cashflow_results(s.get("cashflow_results", {}))
    }
    
    data_to_save = sanitize_data(raw_data)
    
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

def load_project(project_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if not file_path.exists():
        return
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    production_cases = data.get("production_cases", {})
    for case in production_cases.values():
        if "profiles" in case:
            for prof_type in ["gas", "oil", "drilling_plan"]:
                if prof_type in case["profiles"]:
                    case["profiles"][prof_type] = {int(k): v for k, v in case["profiles"][prof_type].items()}
    
    price_cases = data.get("price_cases", {})
    for case in price_cases.values():
        for price_type in ["oil", "gas"]:
            if price_type in case:
                case[price_type] = {int(k): v for k, v in case[price_type].items()}
                
    s = app.storage.user
    s["production_cases"] = production_cases
    s["development_cases"] = deserialize_dev_cases(data.get("development_cases", {}))
    s["price_cases"] = price_cases
    s["cashflow_results"] = deserialize_cashflow_results(data.get("cashflow_results", {}))
    s["current_project"] = project_name

def list_projects():
    return [f.stem for f in DATA_DIR.glob("*.json") if f.stem != "defaults"]

def delete_project(project_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if file_path.exists():
        os.remove(file_path)
        s = app.storage.user
        if s.get("current_project") == project_name:
            s["current_project"] = None
            s["production_cases"] = {}
            s["development_cases"] = {}
            s["price_cases"] = {}
            s["cashflow_results"] = {}
        return True
    return False

def delete_case(project_name: str, case_type: str, case_name: str):
    file_path = DATA_DIR / f"{project_name}.json"
    if not file_path.exists():
        return False
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    type_map = {
        "Production": "production_cases",
        "Development": "development_cases",
        "Price": "price_cases",
        "Cash Flow Result": "cashflow_results"
    }
    
    key = type_map.get(case_type)
    if key and key in data and case_name in data[key]:
        del data[key][case_name]
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        
        s = app.storage.user
        if s.get("current_project") == project_name:
            if key in s and case_name in s[key]:
                temp = s[key]
                del temp[case_name]
                s[key] = temp # Trigger update
        
        return True
    return False

def serialize_dev_cases(dev_cases):
    serialized = {}
    for name, case in dev_cases.items():
        new_case = case.copy()
        if "dev_obj" in new_case:
            dev_obj = new_case["dev_obj"]
            new_case["dev_params_info"] = {
                "dev_start_year": dev_obj.dev_start_year,
                "dev_param": dev_obj.dev_param,
                "development_case": dev_obj.development_case,
                "drill_start_year": dev_obj.drill_start_year,
                "yearly_drilling_schedule": dev_obj.yearly_drilling_schedule,
                "annual_gas_production": dev_obj.annual_gas_production,
                "annual_oil_production": dev_obj.annual_oil_production,
                "annual_capex": dev_obj.annual_capex,
                "annual_opex": dev_obj.annual_opex,
                "annual_abex": dev_obj.annual_abex,
                "total_annual_costs": dev_obj.total_annual_costs,
                "cost_years": dev_obj.cost_years
            }
            del new_case["dev_obj"]
        serialized[name] = new_case
    return serialized

def deserialize_dev_cases(serialized_cases):
    import sys
    # Add core to path if needed, though it should be available in cashflow_nicegui/core
    sys.path.append(str(Path(__file__).parent / "core"))
    from development import DevelopmentCost
    
    deserialized = {}
    for name, case in serialized_cases.items():
        new_case = case.copy()
        
        if "prod_profiles" in new_case:
            for prof_type in ["gas", "oil", "drilling_plan"]:
                if prof_type in new_case["prod_profiles"]:
                    new_case["prod_profiles"][prof_type] = {int(k): v for k, v in new_case["prod_profiles"][prof_type].items()}

        if "dev_params_info" in new_case:
            info = new_case["dev_params_info"]
            dev_obj = DevelopmentCost(
                dev_start_year=info["dev_start_year"],
                dev_param=info["dev_param"],
                development_case=info["development_case"]
            )
            schedule = {int(k): v for k, v in info.get("yearly_drilling_schedule", {}).items()}
            if schedule:
                dev_obj.set_drilling_schedule(
                    drill_start_year=info.get("drill_start_year", info["dev_start_year"]),
                    yearly_drilling_schedule=schedule,
                    already_shifted=True,
                    output=False
                )
            
            dev_obj.set_annual_production(
                annual_gas_production={int(k): v for k, v in info["annual_gas_production"].items()},
                annual_oil_production={int(k): v for k, v in info["annual_oil_production"].items()},
                already_shifted=True,
                output=False
            )
            
            if "annual_capex" in info and info["annual_capex"]:
                dev_obj.annual_capex = {int(k): float(v) for k, v in info["annual_capex"].items()}
                dev_obj.annual_opex = {int(k): float(v) for k, v in info["annual_opex"].items()}
                dev_obj.annual_abex = {int(k): float(v) for k, v in info["annual_abex"].items()}
                dev_obj.total_annual_costs = {int(k): float(v) for k, v in info["total_annual_costs"].items()}
                dev_obj.cost_years = [int(y) for y in info.get("cost_years", sorted(list(dev_obj.annual_capex.keys())))]
                
                cum = 0.0
                dev_obj.cumulative_costs = {}
                for y in sorted(dev_obj.total_annual_costs.keys()):
                    cum += dev_obj.total_annual_costs[y]
                    dev_obj.cumulative_costs[y] = cum
            
            if schedule and not ("annual_capex" in info and info["annual_capex"]):
                dev_obj.calculate_total_costs(output=False)
            
            new_case["dev_obj"] = dev_obj
            del new_case["dev_params_info"]
        deserialized[name] = new_case
    return deserialized

def serialize_cashflow_results(results):
    from pydantic import BaseModel
    serialized = {}
    for name, item in results.items():
        entry = {'inputs': item.get('inputs', {})}
        if 'cf' in item:
            cf = item['cf']
            if hasattr(cf, 'model_dump'):
                 cf_dict = cf.model_dump()
            else:
                 cf_dict = cf.dict()
            
            if 'development_cost' in cf_dict:
                del cf_dict['development_cost']
            entry['cf_data'] = cf_dict
        
        if 'result_summary' in item:
            entry['result_summary'] = item['result_summary']
        serialized[name] = entry
    return serialized

def deserialize_cashflow_results(data):
    import sys
    sys.path.append(str(Path(__file__).parent / "core"))
    from cashflow import CashFlowKOR
    results = {}
    for name, item in data.items():
        cf_data = item.get('cf_data')
        inputs = item.get('inputs', {})
        try:
            res_item = {'inputs': inputs}
            if cf_data:
                cf = CashFlowKOR(**cf_data)
                res_item['cf'] = cf
            if 'result_summary' in item:
                res_item['result_summary'] = item['result_summary']
            if 'cf' in res_item or 'result_summary' in res_item:
                results[name] = res_item
        except Exception as e:
            print(f"Error deserializing cashflow result '{name}': {e}")
    return results

def rounding_decorator(func):
    def wrapper(self, *args, **kwargs) -> Dict[int, float]:
        price_by_year = func(self, *args, **kwargs)
        return {year: round(price, 2) for year, price in price_by_year.items()}
    return wrapper

class PriceDeck:
    def __init__(self,
         start_year: int = 2025, end_year: int = 2075,
         oil_price_initial: float = 70.0, gas_price_initial: float = 8.0, inflation_rate: float = 0.015,
         flat_after_year: int = None,
         oil_price_by_year: Dict[int, float] = None,
         gas_price_by_year: Dict[int, float] = None):

        self.start_year = start_year
        self.end_year = end_year
        self.inflation_rate = inflation_rate
        self.flat_after_year = flat_after_year
        self.years: List[int] = range(start_year, end_year+1, 1)
        self.oil_price_by_year = self._setting_initial_oil_price(oil_price_initial)
        self.gas_price_by_year = self._setting_initial_gas_price(gas_price_initial)

        if oil_price_by_year:
            for y, price in oil_price_by_year.items():
                self.oil_price_by_year[y] = price

        if gas_price_by_year:
            for y, price in gas_price_by_year.items():
                self.gas_price_by_year[y] = price

        if flat_after_year:
            self._setting_flat_price(flat_after_year)

    @rounding_decorator
    def _setting_initial_oil_price(self, oil_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = oil_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    @rounding_decorator
    def _setting_initial_gas_price(self, gas_price_initial) -> Dict[int, float]:
        calculated_prices = {}
        for y in self.years:
            years_from_base = (y - self.start_year)
            calculated_prices[y] = gas_price_initial * ((1 + self.inflation_rate) ** years_from_base)
        return calculated_prices

    def _setting_flat_price(self, flat_after_year: int=None):
        for y in self.years:
            if y > flat_after_year:
                self.oil_price_by_year[y] = self.oil_price_by_year.get(flat_after_year, self.oil_price_by_year[max(self.oil_price_by_year.keys())])
                self.gas_price_by_year[y] = self.gas_price_by_year.get(flat_after_year, self.gas_price_by_year[max(self.gas_price_by_year.keys())])
        return self.oil_price_by_year, self.gas_price_by_year

def get_all_projects_summary() -> pd.DataFrame:
    rows = []
    projects = list_projects()
    for proj_name in projects:
        file_path = DATA_DIR / f"{proj_name}.json"
        row = {
            "Project Name": proj_name,
            "Gas Reserves (BCF)": "N/A",
            "Oil Reserves (MMbbl)": "N/A",
            "Total Revenue ($MM)": "N/A",
            "Total CAPEX ($MM)": "N/A",
            "Net Cash Flow ($MM)": "N/A",
            "NPV (Discounted) ($MM)": "N/A",
            "IRR (%)": "N/A"
        }
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            cf_results = data.get("cashflow_results", {})
            if cf_results:
                first_key = list(cf_results.keys())[0]
                res_item = cf_results[first_key]
                summary = res_item.get("result_summary", {})
                inputs = res_item.get("inputs", {})
                if summary:
                    row["Total Revenue ($MM)"] = f"{summary.get('total_revenue', 0):,.0f}"
                    row["Total CAPEX ($MM)"] = f"{summary.get('total_capex', 0):,.0f}"
                    row["Net Cash Flow ($MM)"] = f"{summary.get('final_cumulative', 0):,.0f}"
                    row["NPV (Discounted) ($MM)"] = f"{summary.get('npv', 0):,.0f}"
                    irr_val = summary.get('irr', 'N/A')
                    if isinstance(irr_val, (int, float)):
                        row["IRR (%)"] = f"{irr_val*100:.1f}%"
                dev_name = inputs.get("dev_name")
                if dev_name:
                    dev_cases = data.get("development_cases", {})
                    dev_data = dev_cases.get(dev_name, {})
                    profiles = dev_data.get("profiles", {})
                    gas_prof = profiles.get("gas", {})
                    oil_prof = profiles.get("oil", {})
                    if not gas_prof and "dev_params_info" in dev_data:
                        gas_prof = dev_data["dev_params_info"].get("annual_gas_production", {})
                        oil_prof = dev_data["dev_params_info"].get("annual_oil_production", {})
                    if gas_prof:
                        total_gas = sum(float(v) for v in gas_prof.values())
                        row["Gas Reserves (BCF)"] = f"{total_gas:,.1f}"
                    if oil_prof:
                        total_oil = sum(float(v) for v in oil_prof.values())
                        row["Oil Reserves (MMbbl)"] = f"{total_oil:,.1f}"
        except Exception as e:
            print(f"Error reading project {proj_name}: {e}")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "Project Name", "Gas Reserves (BCF)", "Oil Reserves (MMbbl)", 
            "Total Revenue ($MM)", "Total CAPEX ($MM)", "Net Cash Flow ($MM)", 
            "NPV (Discounted) ($MM)", "IRR (%)"
        ])
    return pd.DataFrame(rows)
