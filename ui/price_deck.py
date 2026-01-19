from nicegui import ui, app
import pandas as pd
import numpy as np
from utils import ensure_state_init, save_project, list_projects
import plotly.express as px

def content():
    ensure_state_init()
    s = app.storage.user
    
    ui.label("📈 Oil & Gas Price Setup").classes('text-3xl font-bold mb-4')

    # Ensure price decks exist in storage
    if "price_deck_oil" not in s: s["price_deck_oil"] = {y: 70.0 for y in range(2025, 2076)}
    if "price_deck_gas" not in s: s["price_deck_gas"] = {y: 8.0 for y in range(2025, 2076)}

    # --- Case Management ---
    with ui.expansion("📁 Case Management", icon='folder').classes('w-full border rounded mb-4'):
        with ui.row().classes('w-full items-end'):
            existing_cases = list(s.get("price_cases", {}).keys())
            case_selector = ui.select(["Select a case..."] + existing_cases, label="Load Saved Case", value="Select a case...").classes('flex-grow')
            
            def load_case_callback():
                if case_selector.value != "Select a case...":
                    case_data = s["price_cases"][case_selector.value]
                    if "oil" in case_data: s["price_deck_oil"] = {int(k): v for k, v in case_data["oil"].items()}
                    if "gas" in case_data: s["price_deck_gas"] = {int(k): v for k, v in case_data["gas"].items()}
                    if "input_params" in case_data:
                        for k, v in case_data["input_params"].items():
                            s[k] = v
                    ui.notify(f"Price Case '{case_selector.value}' loaded!")
                    ui.navigate.to('/price_deck')
            
            ui.button("📂 Load Selected Case", on_click=load_case_callback).classes('ml-2')
            
        ui.separator().classes('my-4')
        
        with ui.row().classes('w-full items-end'):
            new_case_name = ui.input("New Case Name", value="Base Price").classes('flex-grow')
            
            def save_case_callback():
                if not s.get("current_project"):
                    ui.notify("⚠️ No active project!", type='negative')
                    return
                
                input_params = {k: s.get(k) for k in [
                    "m_start", "m_end", "m_policy", "m_inf",
                    "price_start_year_input", "price_end_year_input",
                    "oil_init_input", "gas_init_input", "price_inflation_input",
                    "price_cap_2x_input"
                ]}
                
                price_case = {
                    "oil": s["price_deck_oil"],
                    "gas": s["price_deck_gas"],
                    "input_params": input_params,
                    "params": {
                        "source": "NiceGUI Enhanced",
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    }
                }
                temp = s.get("price_cases", {})
                temp[new_case_name.value] = price_case
                s["price_cases"] = temp
                save_project(s["current_project"])
                ui.notify(f"Price scenario '{new_case_name.value}' saved!")

            ui.button("💾 Save Current Case", on_click=save_case_callback).classes('ml-2')

    # --- Manual Input Section ---
    ui.label("1. Manual Price Input").classes('text-xl font-bold mt-6 mb-2')
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full items-start'):
            m_start = ui.number("Input Start Year", value=s.get("m_start", 2025), precision=0).bind_value(s, 'm_start')
            m_end = ui.number("Input End Year", value=s.get("m_end", 2035), precision=0).bind_value(s, 'm_end')
            m_policy = ui.select(["Same as end year price", "Apply Inflation"], label="Outbound Policy", value=s.get("m_policy", "Same as end year price")).bind_value(s, 'm_policy').classes('w-48')
            m_inf = ui.number("Outbound Inflation (%)", value=s.get("m_inf", 2.0)).bind_value(s, 'm_inf')

        # We need a table to edit prices
        ui.label("Edit Table (Years as Columns)").classes('text-sm font-semibold mt-4')
        manual_table_container = ui.row().classes('w-full overflow-x-auto p-2 border rounded')
        
        def refresh_manual_table():
            manual_table_container.clear()
            start = int(m_start.value)
            end = int(m_end.value)
            oil = s["price_deck_oil"]
            gas = s["price_deck_gas"]
            
            with ui.column().classes('items-center mr-2'):
                ui.label("Year").classes('font-bold h-10')
                ui.label("Oil ($/bbl)").classes('font-bold h-10')
                ui.label("Gas ($/mcf)").classes('font-bold h-10')
            
            for y in range(start, end + 1):
                with ui.column().classes('items-center'):
                    ui.label(str(y)).classes('h-10 items-center flex')
                    o_val = ui.number(value=oil.get(y, 70.0), format='%.1f').classes('w-20').on_value_change(lambda e, year=y: s["price_deck_oil"].update({year: e.value}))
                    g_val = ui.number(value=gas.get(y, 8.0), format='%.1f').classes('w-20').on_value_change(lambda e, year=y: s["price_deck_gas"].update({year: e.value}))

        refresh_manual_table()
        m_start.on_value_change(refresh_manual_table)
        m_end.on_value_change(refresh_manual_table)

        def apply_manual():
            # Outbound policy handling
            last_year = int(m_end.value)
            oil = s["price_deck_oil"]
            gas = s["price_deck_gas"]
            last_oil = oil[last_year]
            last_gas = gas[last_year]
            inf = m_inf.value / 100.0
            
            temp_oil = oil.copy()
            temp_gas = gas.copy()
            for y in range(last_year + 1, 2076):
                if m_policy.value == "Same as end year price":
                    temp_oil[y] = last_oil
                    temp_gas[y] = last_gas
                else:
                    years_diff = y - last_year
                    temp_oil[y] = last_oil * ((1 + inf) ** years_diff)
                    temp_gas[y] = last_gas * ((1 + inf) ** years_diff)
            
            s["price_deck_oil"] = temp_oil
            s["price_deck_gas"] = temp_gas
            ui.notify("Manual forecast applied!")
            price_plot_container.refresh()

        ui.button("🚀 Apply Manual Forecast", on_click=apply_manual).classes('w-full mt-4').props('primary')

    # --- Automated Price Generation ---
    ui.label("2. Automated Price Generation").classes('text-xl font-bold mt-6 mb-2')
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.grid(columns=3).classes('w-full'):
            a_start = ui.number("Start Year", value=s.get("price_start_year_input"), precision=0).bind_value(s, 'price_start_year_input')
            a_oil_init = ui.number("Initial Oil Price ($/bbl)", value=s.get("oil_init_input")).bind_value(s, 'oil_init_input')
            a_inf = ui.number("Inflation Rate (%)", value=s.get("price_inflation_input")).bind_value(s, 'price_inflation_input')
            
            a_end = ui.number("End Year", value=s.get("price_end_year_input"), precision=0).bind_value(s, 'price_end_year_input')
            a_gas_init = ui.number("Initial Gas Price ($/mcf)", value=s.get("gas_init_input")).bind_value(s, 'gas_init_input')
            cap_2x = ui.checkbox("Stop increasing at 2x initial", value=s.get("price_cap_2x_input")).bind_value(s, 'price_cap_2x_input')

        def generate_auto():
            temp_oil = {}
            temp_gas = {}
            inf = a_inf.value / 100.0
            oil_limit = a_oil_init.value * 2 if cap_2x.value else float('inf')
            gas_limit = a_gas_init.value * 2 if cap_2x.value else float('inf')
            
            for y in range(int(a_start.value), int(a_end.value) + 1):
                years_from_base = y - a_start.value
                calc_oil = a_oil_init.value * ((1 + inf) ** years_from_base)
                calc_gas = a_gas_init.value * ((1 + inf) ** years_from_base)
                temp_oil[y] = min(calc_oil, oil_limit)
                temp_gas[y] = min(calc_gas, gas_limit)
                
            s["price_deck_oil"] = temp_oil
            s["price_deck_gas"] = temp_gas
            ui.notify("Automated prices generated!")
            price_plot_container.refresh()

        ui.button("⚡ Generate & Apply Prices", on_click=generate_auto).classes('w-full mt-4').props('primary')

    # --- Visualization ---
    ui.separator().classes('my-6')
    @ui.refreshable
    def price_plot_container():
        ui.label("Current Price Forecast").classes('text-xl font-bold mb-4')
        oil = s.get("price_deck_oil", {})
        gas = s.get("price_deck_gas", {})
        years = sorted([int(k) for k in oil.keys()])
        if years:
            df = pd.DataFrame({
                'Year': years,
                'Oil Price ($/bbl)': [oil.get(y, 0.0) for y in years],
                'Gas Price ($/mcf)': [gas.get(y, 0.0) for y in years]
            })
            fig = px.line(df, x='Year', y=['Oil Price ($/bbl)', 'Gas Price ($/mcf)'], title="Commodity Price Deck")
            ui.plotly(fig).classes('w-full')
            
            ui.label("Data Summary").classes('text-lg font-bold mt-4')
            ui.aggrid.from_pandas(df).classes('w-full h-64')

    price_plot_container()
