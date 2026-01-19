from nicegui import ui, app
from utils import ensure_state_init, save_defaults, list_projects, delete_project, delete_case, save_project
import pandas as pd
import json
from pathlib import Path

def content():
    ensure_state_init()
    defaults = app.storage.user.get("defaults")
    if not defaults:
        ui.label("⚠️ Defaults not loaded. Please check data/defaults.json.").classes('text-red-500')
        return

    ui.label("⚙️ Global Settings & Defaults").classes('text-3xl font-bold mb-4')
    
    # --- Project Management Section ---
    ui.label("📁 Project & Case Management").classes('text-xl font-semibold mt-6 mb-2')
    with ui.expansion("🛠️ Manage Projects and Cases", icon='settings').classes('w-full border rounded'):
        with ui.tabs().classes('w-full') as tabs:
            add_tab = ui.tab('➕ Add Project')
            remove_tab = ui.tab('🗑️ Remove Project')
            delete_case_tab = ui.tab('❌ Delete Case')
            
        with ui.tab_panels(tabs, value=add_tab).classes('w-full bg-transparent'):
            with ui.tab_panel(add_tab):
                ui.markdown("#### Create New Project")
                new_p_name = ui.input("Project Name").classes('w-full')
                with ui.row().classes('w-full'):
                    new_p_gas = ui.number(label="Gas Reserves (BCF)", value=0.0, step=10.0).classes('flex-grow')
                    new_p_oil = ui.number(label="Oil Reserves (MMbbl)", value=0.0, step=1.0).classes('flex-grow')
                
                def add_project():
                    name = new_p_name.value
                    if name:
                        if name in list_projects():
                            ui.notify(f"Project '{name}' already exists.", type='negative')
                        else:
                            app.storage.user['current_project'] = name
                            app.storage.user['production_cases'] = {}
                            app.storage.user['development_cases'] = {}
                            app.storage.user['price_cases'] = {}
                            app.storage.user['cashflow_results'] = {}
                            
                            if new_p_gas.value > 0 or new_p_oil.value > 0:
                                app.storage.user['production_cases']["Base Case"] = {
                                    "input_params": {
                                        "giip_input": new_p_gas.value,
                                        "oiip_mmbbl": new_p_oil.value,
                                        "well_eur_bcf": defaults["production"]["well_eur_bcf"],
                                        "qi_input": defaults["production"]["qi_mmcfd"],
                                        "prod_dur_input": defaults["production"]["prod_duration"],
                                        "drilling_rate_input": defaults["production"]["drilling_rate"],
                                        "max_rate_input": defaults["production"]["max_prod_rate"]
                                    },
                                    "profiles": {},
                                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                                }
                            save_project(name)
                            ui.notify(f"Project '{name}' created!")
                            ui.navigate.to('/')
                    else:
                        ui.notify("Please enter a project name.", type='negative')
                        
                ui.button("🚀 Add Project", on_click=add_project).classes('w-full mt-4')

            with ui.tab_panel(remove_tab):
                ui.markdown("#### Delete Entire Project")
                projects = list_projects()
                if projects:
                    p_to_delete = ui.select(projects, label="Select Project to Remove").classes('w-full')
                    async def do_delete():
                        if await ui.confirm(f"Are you sure you want to delete project '{p_to_delete.value}'?"):
                            if delete_project(p_to_delete.value):
                                ui.notify(f"Project '{p_to_delete.value}' deleted.")
                                ui.navigate.to('/')
                            else:
                                ui.notify("Failed to delete project.", type='negative')
                    ui.button("⚠️ Delete Project", on_click=do_delete, color='red').classes('w-full mt-4')
                else:
                    ui.label("No projects found.")

            with ui.tab_panel(delete_case_tab):
                ui.markdown("#### Delete Specific Case")
                projects = list_projects()
                if projects:
                    target_p = ui.select(projects, label="Select Target Project").classes('w-full')
                    c_type = ui.select(["Production", "Development", "Price", "Cash Flow Result"], label="Case Category").classes('w-full')
                    
                    case_select_container = ui.column().classes('w-full')
                    
                    def update_cases():
                        case_select_container.clear()
                        if target_p.value and c_type.value:
                            p_file = Path("./data") / f"{target_p.value}.json"
                            if p_file.exists():
                                with open(p_file, "r") as f:
                                    p_data = json.load(f)
                                type_map = {
                                    "Production": "production_cases",
                                    "Development": "development_cases",
                                    "Price": "price_cases",
                                    "Cash Flow Result": "cashflow_results"
                                }
                                category_key = type_map[c_type.value]
                                cases = list(p_data.get(category_key, {}).keys())
                                if cases:
                                    c_to_delete = ui.select(cases, label="Select Case to Delete").classes('w-full')
                                    async def do_delete_case():
                                        if await ui.confirm(f"Delete case '{c_to_delete.value}' from '{target_p.value}'?"):
                                            if delete_case(target_p.value, c_type.value, c_to_delete.value):
                                                ui.notify("Case deleted.")
                                                update_cases()
                                            else:
                                                ui.notify("Error deleting case.", type='negative')
                                    ui.button("❌ Delete Case", on_click=do_delete_case, color='red').classes('w-full mt-2')
                                else:
                                    ui.label(f"No cases found in {c_type.value}.").classes('mt-2 italic')
                    
                    target_p.on_value_change(update_cases)
                    c_type.on_value_change(update_cases)
                else:
                    ui.label("No projects found.")

    ui.separator().classes('my-8')
    ui.markdown("Edit default parameters used across the application. These values will be used when creating new cases.")

    # --- Defaults Tabs ---
    with ui.tabs().classes('w-full') as defaults_tabs:
        prod_tab = ui.tab('🛢️ Production')
        dev_tab = ui.tab('🛠️ Development')
        price_tab = ui.tab('📈 Price Deck')
        eco_tab = ui.tab('⚖️ Economics')

    with ui.tab_panels(defaults_tabs, value=prod_tab).classes('w-full bg-transparent'):
        with ui.tab_panel(prod_tab):
            ui.label("Field & Production Defaults").classes('text-xl font-semibold mb-4')
            with ui.grid(columns=2).classes('w-full'):
                qi = ui.number("Initial Rate (MMcf/d)", value=defaults["production"]["qi_mmcfd"], step=1.0)
                giip = ui.number("Gas Reserves (BCF)", value=defaults["production"]["giip_bcf"], step=10.0)
                well_eur = ui.number("Well EUR (BCF)", value=defaults["production"]["well_eur_bcf"], step=0.5)
                oiip = ui.number("Oil Reserves (MMbbl)", value=defaults["production"]["oiip_mmbbl"], step=1.0)
                prod_dur = ui.number("Prod. Period (Years)", value=defaults["production"]["prod_duration"], precision=0)
                drill_rate = ui.number("Drilling Rate (Wells/Year)", value=defaults["production"]["drilling_rate"], precision=0)
                max_rate = ui.number("Max Prod. Rate (MMcf/y)", value=defaults["production"]["max_prod_rate"], step=1000)

        with ui.tab_panel(dev_tab):
            ui.label("Development Cost & Timing Defaults").classes('text-xl font-semibold mb-4')
            with ui.grid(columns=2).classes('w-full'):
                sunk = ui.number("Sunk Cost (MM$)", value=defaults["development"]["sunk_cost"])
                exp_start = ui.number("Exploration Start Year", value=defaults["development"]["exp_start_year"], precision=0)
                dev_start = ui.number("Development Start Year", value=defaults["development"]["dev_start_year"], precision=0)
                drill_start = ui.number("Drilling Start Year", value=defaults["development"]["drill_start_year"], precision=0)
            
            ui.separator().classes('my-4')
            ui.label("Study & PM Costs").classes('text-lg font-semibold mb-2')
            with ui.grid(columns=3).classes('w-full'):
                feas_c = ui.number("Feasibility Cost", value=defaults["development"]["study_costs"]["feasibility"]["cost"])
                feas_t = ui.number("Feasibility Timing", value=defaults["development"]["study_costs"]["feasibility"]["timing"], precision=0)
                pm_w = ui.number("PM & Others (per well)", value=defaults["development"]["study_costs"]["pm_others_per_well"])
                conc_c = ui.number("Concept Cost", value=defaults["development"]["study_costs"]["concept"]["cost"])
                conc_t = ui.number("Concept Timing", value=defaults["development"]["study_costs"]["concept"]["timing"], precision=0)

            ui.separator().classes('my-4')
            ui.label("Facility CAPEX").classes('text-lg font-semibold mb-2')
            with ui.grid(columns=2).classes('w-full'):
                drill_pw = ui.number("Drilling Cost per Well", value=defaults["development"]["facility_capex"]["drilling_per_well"])
                fpso_c = ui.number("FPSO Cost", value=defaults["development"]["facility_capex"]["fpso_fpso"]["cost"])
                subsea_pw = ui.number("Subsea Cost per Well", value=defaults["development"]["facility_capex"]["subsea_per_well"])
                fpso_d = ui.number("FPSO Duration", value=defaults["development"]["facility_capex"]["fpso_fpso"]["duration"], precision=0)

            ui.separator().classes('my-4')
            ui.label("OPEX & ABEX").classes('text-lg font-semibold mb-2')
            with ui.grid(columns=2).classes('w-full'):
                opex_bcf = ui.number("OPEX per BCF", value=defaults["development"]["opex_abex"]["opex_per_bcf"], step=0.001, format='%.3f')
                abex_pw = ui.number("ABEX per Well", value=defaults["development"]["opex_abex"]["abex_per_well"])
                opex_f = ui.number("OPEX Fixed (MM$/y)", value=defaults["development"]["opex_abex"]["opex_fixed"])
                abex_f = ui.number("ABEX FPSO", value=defaults["development"]["opex_abex"]["abex_fpso_fpso"])

        with ui.tab_panel(price_tab):
            ui.label("Price Deck Defaults").classes('text-xl font-semibold mb-4')
            with ui.grid(columns=2).classes('w-full'):
                o_init = ui.number("Initial Oil Price ($/bbl)", value=defaults["price_deck"]["oil_init"])
                p_start = ui.number("Price Start Year", value=defaults["price_deck"]["start_year"], precision=0)
                g_init = ui.number("Initial Gas Price ($/mcf)", value=defaults["price_deck"]["gas_init"])
                p_end = ui.number("Price End Year", value=defaults["price_deck"]["end_year"], precision=0)
                p_inf = ui.number("Price Inflation (%)", value=defaults["price_deck"]["inflation_pct"])
                p_cap = ui.checkbox("Cap at 2x Initial Price", value=defaults["price_deck"]["cap_2x"])

        with ui.tab_panel(eco_tab):
            ui.label("Global Economic Defaults").classes('text-xl font-semibold mb-4')
            with ui.grid(columns=2).classes('w-full'):
                b_year = ui.number("Base Year (for PV)", value=defaults["economics"]["base_year"], precision=0)
                c_inf = ui.number("Cost Inflation Rate", value=defaults["economics"]["cost_inflation"], step=0.001, format='%.4f')
                d_rate = ui.number("Discount Rate (fraction)", value=defaults["economics"]["discount_rate"], step=0.01, format='%.2f')
                dep_method = ui.select(["Unit of Production", "Straight Line", "Declining Balance"], label="Depreciation Method", value=defaults["economics"]["depreciation_method"]).classes('w-full')
                ex_rate = ui.number("Exchange Rate (KRW/USD)", value=defaults["economics"]["exchange_rate"])
                u_life = ui.number("Useful Life (Years)", value=defaults["economics"]["useful_life"], precision=0)

    def save_settings():
        new_defaults = defaults.copy()
        new_defaults["production"].update({
            "qi_mmcfd": qi.value, "giip_bcf": giip.value, "well_eur_bcf": well_eur.value,
            "oiip_mmbbl": oiip.value, "prod_duration": int(prod_dur.value),
            "drilling_rate": int(drill_rate.value), "max_prod_rate": int(max_rate.value)
        })
        new_defaults["development"].update({
            "sunk_cost": sunk.value, "exp_start_year": int(exp_start.value),
            "dev_start_year": int(dev_start.value), "drill_start_year": int(drill_start.value)
        })
        new_defaults["development"]["study_costs"].update({
            "feasibility": {"cost": feas_c.value, "timing": int(feas_t.value), "duration": defaults["development"]["study_costs"]["feasibility"]["duration"]},
            "concept": {"cost": conc_c.value, "timing": int(conc_t.value), "duration": defaults["development"]["study_costs"]["concept"]["duration"]},
            "pm_others_per_well": pm_w.value
        })
        new_defaults["development"]["facility_capex"].update({
            "drilling_per_well": drill_pw.value, "subsea_per_well": subsea_pw.value,
            "fpso_fpso": {"cost": fpso_c.value, "timing": defaults["development"]["facility_capex"]["fpso_fpso"]["timing"], "duration": int(fpso_d.value)},
            "pipeline_fpso": defaults["development"]["facility_capex"]["pipeline_fpso"],
            "terminal_fpso": defaults["development"]["facility_capex"]["terminal_fpso"]
        })
        new_defaults["development"]["opex_abex"].update({
            "opex_per_bcf": opex_bcf.value, "opex_fixed": opex_f.value,
            "abex_per_well": abex_pw.value, "abex_fpso_fpso": abex_f.value
        })
        new_defaults["price_deck"].update({
            "oil_init": o_init.value, "gas_init": g_init.value, "inflation_pct": p_inf.value,
            "start_year": int(p_start.value), "end_year": int(p_end.value), "cap_2x": p_cap.value
        })
        new_defaults["economics"].update({
            "base_year": int(b_year.value), "discount_rate": d_rate.value, "exchange_rate": ex_rate.value,
            "cost_inflation": c_inf.value, "depreciation_method": dep_method.value, "useful_life": int(u_life.value)
        })
        
        save_defaults(new_defaults)
        app.storage.user["defaults"] = new_defaults
        ui.notify("✅ Settings saved successfully!", type='positive')

    ui.button("💾 Save Settings", on_click=save_settings).classes('w-full mt-8').props('primary')
