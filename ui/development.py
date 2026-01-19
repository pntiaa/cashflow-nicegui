from nicegui import ui, app
import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path
from utils import ensure_state_init, save_project, list_projects
from production import YearlyProductionProfile
from development import DevelopmentCost
import plotly.express as px
from plotting import plot_dev_cost_profile, plot_detailed_cost_breakdown, plot_df_profile

def content():
    ensure_state_init()
    s = app.storage.user
    defaults = s.get("defaults")
    
    ui.label("Development & Production Setup").classes('text-3xl font-bold mb-4')
    ui.markdown("자원량을 기준으로 시추공수를 계산하고, 그에 따른 개발비를 계산합니다.")

    # --- Case Management ---
    with ui.expansion("📁 Case Management", icon='folder').classes('w-full border rounded mb-4'):
        with ui.row().classes('w-full items-end'):
            existing_cases = list(s.get("development_cases", {}).keys())
            case_selector = ui.select(["Select a case..."] + existing_cases, label="Load Saved Case", value="Select a case...").classes('flex-grow')
            
            def load_case_callback():
                if case_selector.value != "Select a case...":
                    case_data = s["development_cases"][case_selector.value]
                    # Restore inputs to storage
                    if "input_params" in case_data:
                        for k, v in case_data["input_params"].items():
                            s[k] = v
                    # Restore results
                    if "dev_obj" in case_data:
                        s["current_dev_obj"] = case_data["dev_obj"]
                        s["dev_results_ready"] = True
                    if "profiles" in case_data:
                        prof = case_data["profiles"]
                        s["prod_data_dict"] = prof
                        s["drilling_plan_results"] = prof.get("drilling_plan", {})
                    ui.notify(f"Case '{case_selector.value}' loaded!")
                    ui.navigate.to('/development') # Refresh
            
            ui.button("📂 Load Selected Case", on_click=load_case_callback).classes('ml-2')
            
        ui.separator().classes('my-4')
        
        with ui.row().classes('w-full items-end'):
            new_case_name = ui.input("New Case Name", value="Base Case").classes('flex-grow')
            
            def save_case_callback():
                if not s.get("current_project"):
                    ui.notify("⚠️ No active project!", type='negative')
                    return
                if not s.get("dev_results_ready"):
                    ui.notify("⚠️ Please calculate results before saving.", type='negative')
                    return
                
                # Collect inputs
                input_params = {k: s.get(k) for k in [
                    "qi_input", "well_eur_input", "prod_dur_input", "giip_input", "oiip_input",
                    "drilling_rate_input", "max_rate_input", "sunk_cost_input", "exp_start_year_input",
                    "dev_start_year_input", "drill_start_year_input", "dev_case_input",
                    "feas_study_input", "feas_study_t", "feas_study_d",
                    "concept_study_input", "concept_study_t", "concept_study_d",
                    "feed_cost_input", "feed_cost_t", "feed_cost_d",
                    "pm_others_input", "drilling_cost_input", "subsea_cost_input",
                    "fpso_cost_input", "fpso_cost_t", "fpso_cost_d",
                    "pipeline_cost_input", "pipeline_cost_t", "pipeline_cost_d",
                    "opex_per_bcf_input", "opex_fixed_input", "abex_per_well_input", "abex_fpso_input"
                ]}
                
                dev = s["current_dev_obj"]
                case_data = {
                    "input_params": input_params,
                    "cost_summary": {
                        "total_capex": dev.total_capex,
                        "total_opex": dev.total_opex,
                        "total_abex": dev.total_abex
                    },
                    "profiles": s.get("prod_data_dict", {}),
                    "dev_obj": dev
                }
                temp = s["development_cases"]
                temp[new_case_name.value] = case_data
                s["development_cases"] = temp
                save_project(s["current_project"])
                ui.notify(f"Case '{new_case_name.value}' saved!")

            ui.button("💾 Save Current Case", on_click=save_case_callback).classes('ml-2')

    # --- Production Profile Section ---
    ui.label("🛢️ Production Profile Generation").classes('text-xl font-bold mt-6 mb-2')
    with ui.card().classes('w-full mb-6'):
        with ui.row().classes('w-full'):
            # Left: Type Curve
            with ui.column().classes('w-1/3 p-4 border-r'):
                ui.label("Type Curve Setup").classes('font-bold mb-2')
                qi = ui.number("Initial Rate (MMcf/d)", value=s.get("qi_input"), step=1.0).bind_value(s, 'qi_input').classes('w-full')
                well_eur = ui.number("Well EUR (BCF)", value=s.get("well_eur_input"), step=1.0).bind_value(s, 'well_eur_input').classes('w-full')
                
                def generate_tc():
                    dur = int(s.get("prod_dur_input", 30))
                    profile = YearlyProductionProfile(production_duration=dur)
                    profile.generate_type_curve_from_exponential(
                        qi_mmcfd=qi.value,
                        EUR_target_mmcf=well_eur.value * 1000,
                        T_years=dur
                    )
                    s["profile_obj"] = profile # Not serializable directly, but for current session
                    s["tc_data_dict"] = {
                        'Year': list(range(1, len(profile.yearly_type_rate) + 1)),
                        'Rate': [float(v) for v in profile.yearly_type_rate]
                    }
                    ui.notify("Type Curve Generated!")
                    tc_plot_container.refresh()

                ui.button("🚀 Generate Type Curve", on_click=generate_tc).classes('w-full mt-4')
                
                @ui.refreshable
                def tc_plot_container():
                    data = s.get("tc_data_dict")
                    if data:
                        fig = px.line(x=data['Year'], y=data['Rate'], title="Annual Rate vs. Years")
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        ui.plotly(fig).classes('w-full')
                
                tc_plot_container()

            # Right: Field Profile
            with ui.column().classes('w-2/3 p-4'):
                ui.label("Field Profile Setup").classes('font-bold mb-2')
                with ui.grid(columns=3).classes('w-full'):
                    giip = ui.number("Gas Reserves (BCF)", value=s.get("giip_input"), step=100.0).bind_value(s, 'giip_input')
                    oiip = ui.number("Oil Reserves (MMbbl)", value=s.get("oiip_input"), step=10.0).bind_value(s, 'oiip_input')
                    prod_dur = ui.number("Prod. Period (Years)", value=s.get("prod_dur_input"), precision=0).bind_value(s, 'prod_dur_input')
                    drill_rate = ui.number("Drilling Rate (Wells/Year)", value=s.get("drilling_rate_input"), precision=0).bind_value(s, 'drilling_rate_input')
                    max_rate = ui.number("Max Prod. Rate (MMcf/y)", value=s.get("max_rate_input")).bind_value(s, 'max_rate_input')
                
                def generate_field_profile():
                    if not s.get("profile_obj"):
                        ui.notify("⚠️ Please generate a Type Curve first.", type='warning')
                        return
                    
                    profile = s["profile_obj"]
                    profile.production_duration = int(prod_dur.value)
                    wells_to_drill = math.ceil(giip.value / well_eur.value)
                    s["wells_to_drill"] = wells_to_drill
                    
                    drilling_plan = profile.make_drilling_plan(total_wells_number=wells_to_drill, drilling_rate=drill_rate.value)
                    gas_profile = profile.make_production_profile_yearly(peak_production_annual=max_rate.value if max_rate.value > 0 else None)
                    cgr = (oiip.value / giip.value) * 1000
                    oil_profile = {year: gas * cgr / 1000 for year, gas in gas_profile.items()}
                    drilling_plan = {year: drilling_plan.get(year, 0) for year in range(int(prod_dur.value))}
                    
                    s["prod_data_dict"] = {
                        "gas": gas_profile,
                        "oil": oil_profile,
                        "drilling_plan": drilling_plan
                    }
                    s["drilling_plan_results"] = drilling_plan
                    ui.notify("Field Production Profile Generated!")
                    field_plot_container.refresh()

                ui.button("🚀 Generate Field Production Profile", on_click=generate_field_profile).classes('w-full mt-4')
                
                @ui.refreshable
                def field_plot_container():
                    data = s.get("prod_data_dict")
                    if data:
                        years = list(data["gas"].keys())
                        gas = list(data["gas"].values())
                        oil = list(data["oil"].values())
                        wells = list(data["drilling_plan"].values())
                        fig = plot_df_profile(years, gas, oil, wells)
                        fig.update_layout(height=400)
                        ui.plotly(fig).classes('w-full')
                        ui.label(f"🔢 Estimated Total Wells: {s.get('wells_to_drill')}").classes('italic text-slate-500')
                
                field_plot_container()

    # --- Development Cost Section ---
    ui.label("🛠️ Development Cost Generation").classes('text-xl font-bold mt-6 mb-2')
    with ui.card().classes('w-full p-4 mb-6'):
        # Exploration Costs
        ui.label("🔍 Exploration Costs").classes('font-bold mb-2')
        with ui.row().classes('items-center mb-4'):
            sunk = ui.number("Sunk Cost", value=s.get("sunk_cost_input")).bind_value(s, 'sunk_cost_input')
            exp_start = ui.number("Exploration Start Year", value=s.get("exp_start_year_input"), precision=0).bind_value(s, 'exp_start_year_input')
        
        # We need a way to edit exploration costs by year.
        # For now, let's use a simple list or a grid of inputs.
        ui.label("Exploration Costs by Year (MM$)").classes('text-sm font-semibold')
        exp_container = ui.row().classes('w-full overflow-x-auto')
        
        def refresh_exp_inputs():
            exp_container.clear()
            start = int(s.get("exp_start_year_input", 2020))
            if "exploration_costs" not in s:
                s["exploration_costs"] = {y: 0.0 for y in range(start, start + 5)}
            
            # Ensure we have the current years
            costs = s["exploration_costs"]
            for y in range(start, start + 5):
                val = costs.get(y, 0.0)
                ui.number(str(y), value=val, format='%.1f').classes('w-24').on_value_change(lambda e, year=y: s["exploration_costs"].update({year: e.value}))
        
        refresh_exp_inputs()
        exp_start.on_value_change(refresh_exp_inputs)

        ui.separator().classes('my-4')
        
        # Development Timing and Case
        with ui.row().classes('w-full items-start'):
            dev_start = ui.number("Development Start Year", value=s.get("dev_start_year_input"), precision=0).bind_value(s, 'dev_start_year_input')
            drill_start = ui.number("Drilling Start Year", value=s.get("drill_start_year_input"), precision=0).bind_value(s, 'drill_start_year_input')
            with ui.column():
                ui.label("Development Case").classes('text-sm font-semibold')
                dev_case = ui.radio(["FPSO_case", "tie-back_case"], value=s.get("dev_case_input")).bind_value(s, 'dev_case_input')

        ui.separator().classes('my-4')
        
        # Study & PM Costs
        ui.label("📋 Study & PM Costs").classes('font-bold mb-2')
        with ui.grid(columns=4).classes('w-full items-end'):
            feas_c = ui.number("Feasibility Cost", value=s.get("feas_study_input")).bind_value(s, 'feas_study_input')
            feas_t = ui.number("Timing", value=s.get("feas_study_t")).bind_value(s, 'feas_study_t')
            feas_d = ui.number("Duration", value=s.get("feas_study_d")).bind_value(s, 'feas_study_d')
            pm_w = ui.number("PM & Others (per Well)", value=s.get("pm_others_input")).bind_value(s, 'pm_others_input')
            
            conc_c = ui.number("Concept Cost", value=s.get("concept_study_input")).bind_value(s, 'concept_study_input')
            conc_t = ui.number("Timing", value=s.get("concept_study_t")).bind_value(s, 'concept_study_t')
            conc_d = ui.number("Duration", value=s.get("concept_study_d")).bind_value(s, 'concept_study_d')
            
            feed_c = ui.number("FEED Cost", value=s.get("feed_cost_input")).bind_value(s, 'feed_cost_input')
            feed_t = ui.number("Timing", value=s.get("feed_cost_t")).bind_value(s, 'feed_cost_t')
            feed_d = ui.number("Duration", value=s.get("feed_cost_d")).bind_value(s, 'feed_cost_d')

        ui.separator().classes('my-4')
        
        # Facility CAPEX
        ui.label("🏗️ Facility CAPEX").classes('font-bold mb-2')
        with ui.grid(columns=4).classes('w-full items-end'):
            drill_pw = ui.number("Drilling Cost/Well", value=s.get("drilling_cost_input")).bind_value(s, 'drilling_cost_input')
            subsea_pw = ui.number("Subsea Cost/Well", value=s.get("subsea_cost_input")).bind_value(s, 'subsea_cost_input')
            
            fpso_c = ui.number("FPSO Cost", value=s.get("fpso_cost_input")).bind_value(s, 'fpso_cost_input')
            fpso_t = ui.number("Timing", value=s.get("fpso_cost_t")).bind_value(s, 'fpso_cost_t')
            fpso_d = ui.number("Duration", value=s.get("fpso_cost_d")).bind_value(s, 'fpso_cost_d')
            
            pipe_c = ui.number("Pipeline Cost", value=s.get("pipeline_cost_input")).bind_value(s, 'pipeline_cost_input')
            pipe_t = ui.number("Timing", value=s.get("pipeline_cost_t")).bind_value(s, 'pipeline_cost_t')
            pipe_d = ui.number("Duration", value=s.get("pipeline_cost_d")).bind_value(s, 'pipeline_cost_d')

        ui.separator().classes('my-4')
        
        # OPEX & ABEX
        ui.label("💸 OPEX & ABEX").classes('font-bold mb-2')
        with ui.grid(columns=4).classes('w-full items-end'):
            opex_bcf = ui.number("OPEX per BCF", value=s.get("opex_per_bcf_input"), format="%.3f").bind_value(s, 'opex_per_bcf_input')
            opex_f = ui.number("OPEX Fixed ($MM/y)", value=s.get("opex_fixed_input")).bind_value(s, 'opex_fixed_input')
            abex_pw = ui.number("ABEX per Well", value=s.get("abex_per_well_input")).bind_value(s, 'abex_per_well_input')
            abex_f = ui.number("ABEX FPSO", value=s.get("abex_fpso_input")).bind_value(s, 'abex_fpso_input')

        def calculate_costs():
            if not s.get("prod_data_dict"):
                ui.notify("⚠️ Please generate a Production Profile first.", type='warning')
                return
            
            # Pack parameters
            dev_param = {dev_case.value: {
                'drilling_cost': drill_pw.value, 
                'Subsea_cost': subsea_pw.value,
                'feasability_study': {'cost': feas_c.value, 'timing': feas_t.value, 'duration': feas_d.value},
                'concept_study_cost': {'cost': conc_c.value, 'timing': conc_t.value, 'duration': conc_d.value},
                'FEED_cost': {'cost': feed_c.value, 'timing': feed_t.value, 'duration': feed_d.value},
                'PM_others_cost': pm_w.value,
                'EIA_cost': {'cost': 0.0, 'timing': 0, 'duration': 1},
                'FPSO_cost': {'cost': fpso_c.value, 'timing': fpso_t.value, 'duration': fpso_d.value},
                'export_pipeline_cost': {'cost': pipe_c.value, 'timing': pipe_t.value, 'duration': pipe_d.value},
                'terminal_cost': {'cost': 0.0, 'timing': 0, 'duration': 1},
                'OPEX_per_bcf': opex_bcf.value, 'OPEX_fixed': opex_f.value,
                'ABEX_per_well': abex_pw.value, 'ABEX_FPSO': abex_f.value,
                'ABEX_subsea': 14.0, 'ABEX_onshore_pipeline': 0.5, 'ABEX_offshore_pipeline': 11.0
            }}
            
            dev = DevelopmentCost(dev_start_year=int(dev_start.value), dev_param=dev_param, development_case=dev_case.value)
            dev.set_drilling_schedule(drill_start_year=int(drill_start.value), yearly_drilling_schedule=s["drilling_plan_results"])
            
            prod = s["prod_data_dict"]
            dev.set_annual_production(
                annual_gas_production={int(k): v for k, v in prod["gas"].items()},
                annual_oil_production={int(k): v for k, v in prod["oil"].items()}
            )
            
            dev.set_exploration_stage(
                exploration_start_year=int(exp_start.value),
                exploration_costs={int(k): v for k, v in s.get("exploration_costs", {}).items()},
                sunk_cost=sunk.value
            )
            
            dev.calculate_total_costs()
            s["current_dev_obj"] = dev
            s["dev_results_ready"] = True
            ui.notify("Development costs calculated!")
            cost_results_container.refresh()

        ui.button("🔄 Apply Parameters & Calculate", on_click=calculate_costs).classes('w-full mt-4').props('primary')

    # --- Results Visualization ---
    @ui.refreshable
    def cost_results_container():
        if s.get("dev_results_ready"):
            dev = s["current_dev_obj"]
            ui.label("### 📊 Development Cost Profile (MM$)").classes('text-xl font-bold mt-6')
            ui.plotly(plot_dev_cost_profile(dev)).classes('w-full')
            
            ui.label("### 📊 Detailed CAPEX Breakdown (MM$)").classes('text-xl font-bold mt-6')
            ui.plotly(plot_detailed_cost_breakdown(dev)).classes('w-full')
            
            ui.label("### Detailed Cost Table").classes('text-lg font-bold mt-4')
            breakdown = dev.get_cost_breakdown()
            capex_data = breakdown.get('capex_breakdown', {})
            if capex_data:
                df = pd.DataFrame(capex_data).T
                df.columns = df.columns.astype(int)
                df = df.reindex(sorted(df.columns), axis=1)
                # Display as AG Grid or simple table
                ui.aggrid.from_pandas(df.reset_index()).classes('w-full h-96')

    cost_results_container()
