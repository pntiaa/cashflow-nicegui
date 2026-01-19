from nicegui import ui, app
import pandas as pd
import io
import sys
from pathlib import Path
from utils import ensure_state_init, save_project
from cashflow import CashFlowKOR
from plotting import plot_cashflow, plot_cf_sankey_chart, plot_cf_waterfall_chart, plot_production_profile, plot_cost_profile
import plotly.graph_objects as go

def content():
    ensure_state_init()
    s = app.storage.user
    
    ui.label("💰 Cash Flow Analysis").classes('text-3xl font-bold mb-4')

    # --- Dependency Check ---
    missing_deps = []
    if not s.get("development_cases"): missing_deps.append("Development")
    if not s.get("price_cases"): missing_deps.append("Price Deck")
    if not s.get("current_project"): missing_deps.append("Project")
    
    if missing_deps:
        ui.label(f"⚠️ Missing saved cases from: {', '.join(missing_deps)}. Please complete those pages first.").classes('text-orange-500 font-bold')
        return

    # --- Scenario Selection ---
    ui.label("🏁 Run Economic Scenario").classes('text-xl font-bold mt-6 mb-2')
    with ui.row().classes('w-full items-stretch mb-6'):
        with ui.card().classes('flex-grow p-4'):
            dev_name = ui.select(list(s["development_cases"].keys()), label="Select Development Case").classes('w-full')
            
            @ui.refreshable
            def dev_summary():
                if dev_name.value:
                    data = s["development_cases"][dev_name.value]
                    prod = data.get('profiles', {})
                    gas = sum(prod.get("gas", {}).values())
                    oil = sum(prod.get("oil", {}).values())
                    capex = data.get('cost_summary', {}).get("total_capex", 0.0)
                    opex = data.get('cost_summary', {}).get("total_opex", 0.0)
                    
                    ui.label("📋 Case Summary").classes('font-bold mt-2')
                    with ui.row().classes('w-full'):
                        with ui.column():
                            ui.label(f"- Total Gas: {gas:,.1f} BCF")
                            ui.label(f"- Total Oil: {oil:,.1f} MMbbl")
                        with ui.column():
                            ui.label(f"- Total CAPEX: {capex:,.1f} MM$")
                            ui.label(f"- Total OPEX: {opex:,.1f} MM$")
            
            dev_summary()
            dev_name.on_value_change(dev_summary.refresh)

        with ui.card().classes('flex-grow p-4'):
            price_name = ui.select(list(s["price_cases"].keys()), label="Select Price Scenario").classes('w-full')
            
            @ui.refreshable
            def price_plot():
                if price_name.value:
                    data = s["price_cases"][price_name.value]
                    years = sorted([int(k) for k in data['gas'].keys()])
                    oil_y = [data['oil'].get(y, data['oil'].get(str(y))) for y in years]
                    gas_y = [data['gas'].get(y, data['gas'].get(str(y))) for y in years]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=gas_y, mode='lines', name='Gas Price'))
                    fig.add_trace(go.Scatter(x=years, y=oil_y, mode='lines', name='Oil Price'))
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    ui.plotly(fig).classes('w-full')
            
            price_plot()
            price_name.on_value_change(price_plot.refresh)

    # --- Global Economic Inputs ---
    with ui.expansion("⚙️ Global Economic Parameters", icon='settings').classes('w-full border rounded mb-6'):
        with ui.grid(columns=3).classes('w-full p-4'):
            b_year = ui.number("Base Year (for PV)", value=s.get("base_year_input"), precision=0).bind_value(s, "base_year_input")
            d_rate = ui.number("Discount Rate (fraction)", value=s.get("discount_rate_input"), format="%.2f").bind_value(s, "discount_rate_input")
            ex_rate = ui.number("Exchange Rate (KRW/USD)", value=s.get("exchange_rate_input")).bind_value(s, "exchange_rate_input")
            c_inf = ui.number("Cost Inflation Rate", value=s.get("cost_inflation_input"), format="%.4f").bind_value(s, "cost_inflation_input")
            dep_method = ui.select(["Unit of Production", "Straight Line", "Declining Balance"], label="Depreciation Method", value=s.get("depreciation_method_input")).bind_value(s, "depreciation_method_input")
            u_life = ui.number("Useful Life (Depreciation)", value=s.get("useful_life_input"), precision=0).bind_value(s, "useful_life_input")

    def run_analysis():
        if not dev_name.value or not price_name.value:
            ui.notify("Please select both cases.", type='warning')
            return
        
        d_case = s["development_cases"][dev_name.value]
        p_case = s["price_cases"][price_name.value]
        dev_obj = d_case['dev_obj']
        
        cf = CashFlowKOR(
            base_year=int(b_year.value),
            oil_price_by_year={int(k): v for k, v in p_case['oil'].items()},
            gas_price_by_year={int(k): v for k, v in p_case['gas'].items()},
            cost_inflation_rate=c_inf.value,
            discount_rate=d_rate.value,
            exchange_rate=ex_rate.value
        )
        
        cf.set_development_costs(dev_obj, output=False)
        
        def ensure_actual_years(d, start_year):
            if not d: return {}
            min_key = min(int(k) for k in d.keys())
            if min_key < 1000: return {int(k) + start_year: v for k, v in d.items()}
            return {int(k): v for k, v in d.items()}

        oil_shifted = ensure_actual_years(dev_obj.annual_oil_production, dev_obj.drill_start_year)
        gas_shifted = ensure_actual_years(dev_obj.annual_gas_production, dev_obj.drill_start_year)
        cf.set_production_profile_from_dicts(oil_dict=oil_shifted, gas_dict=gas_shifted)
        
        # Run Calculation Steps
        cf.calculate_annual_revenue(output=False)    
        cf.determine_cop_year(output=False)
        cf.apply_cop_adjustments(output=False)
        cf.calculate_royalty(output=False)
        cf.calculate_high_price_royalty(output=False)
        cf.calculate_depreciation(method=dep_method.value.lower().replace(" ", "_"), useful_life=int(u_life.value), output=False)
        cf.calculate_taxes(investment_tax_credit=True, local_tax=True, output=False)
        cf.calculate_net_cash_flow(output=False)
        cf.calculate_npv(output=False)
        
        s["last_cf_result"] = {
            'cf': cf,
            'dev_name': dev_name.value,
            'price_name': price_name.value,
            'dev_obj': dev_obj,
            'project_name': s["current_project"],
            'inputs': {
                'dev_name': dev_name.value,
                'price_name': price_name.value,
                'global_params': {
                    'base_year': b_year.value,
                    'discount_rate': d_rate.value,
                    'exchange_rate': ex_rate.value,
                    'cost_inflation': c_inf.value,
                    'depreciation_method': dep_method.value,
                    'useful_life': u_life.value
                }
            }
        }
        ui.notify("✅ Cash flow calculation complete!")
        results_container.refresh()

    ui.button("🚀 Run Cash Flow Analysis", on_click=run_analysis).classes('w-full mb-8').props('primary')

    @ui.refreshable
    def results_container():
        res = s.get("last_cf_result")
        if not res:
            ui.label("💡 Click the button above to run the economic analysis.").classes('italic text-slate-500')
            return
        
        cf = res['cf']
        summ = cf.get_project_summary()
        
        ui.separator().classes('my-4')
        ui.label("📊 Economic Summary").classes('text-2xl font-bold mb-4')
        
        with ui.row().classes('w-full items-start'):
            with ui.card().classes('w-1/4 p-4'):
                ui.label("Key Metrics").classes('font-bold border-b mb-2')
                with ui.row().classes('justify-between w-full'):
                    ui.label(f"NPV (@{s['discount_rate_input']*100:.1f}%)")
                    ui.label(f"${summ['npv']:,.0f} MM").classes('font-bold')
                with ui.row().classes('justify-between w-full'):
                    ui.label("IRR")
                    ui.label(f"{summ['irr']*100:.1f}%" if isinstance(summ['irr'], (int, float)) else "N/A").classes('font-bold')
                with ui.row().classes('justify-between w-full'):
                    ui.label("Payback Year")
                    ui.label(str(summ['payback_year']) if summ['payback_year'] else "N/A").classes('font-bold')
                with ui.row().classes('justify-between w-full'):
                    ui.label("COP Year")
                    ui.label(str(cf.cop_year) if cf.cop_year else "N/A").classes('font-bold')

            with ui.card().classes('w-1/4 p-4'):
                ui.label("Totals ($MM)").classes('font-bold border-b mb-2')
                for k, v in [("Revenue", summ['total_revenue']), ("CAPEX", summ['total_capex']), 
                             ("OPEX", summ['total_opex']), ("ABEX", summ['total_abex']), 
                             ("Royalty", summ['total_royalty']), ("Tax", summ['total_tax']),
                             ("Net Cash Flow", summ['final_cumulative'])]:
                    with ui.row().classes('justify-between w-full'):
                        ui.label(k)
                        ui.label(f"{v:,.0f}").classes('font-bold')

            with ui.card().classes('flex-grow p-4'):
                ui.label("Cash Flow Distribution").classes('font-bold mb-2')
                fig = plot_cf_waterfall_chart(cf)
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                ui.plotly(fig).classes('w-full')

        ui.label("📈 Detailed Visualizations").classes('text-2xl font-bold mt-8 mb-4')
        with ui.grid(columns=2).classes('w-full'):
            with ui.card().classes('p-4'):
                ui.label("Cash Flow Profile").classes('font-bold mb-2')
                ui.plotly(plot_cashflow(cf)).classes('w-full')
            with ui.card().classes('p-4'):
                ui.label("Production Profile").classes('font-bold mb-2')
                ui.plotly(plot_production_profile(cf)).classes('w-full')
        
        with ui.card().classes('w-full p-4 mt-4'):
            ui.label("Cost Profile").classes('font-bold mb-2')
            ui.plotly(plot_cost_profile(res['dev_obj'])).classes('w-full')

        ui.label("📄 Detailed Annual Cash Flow Table").classes('text-2xl font-bold mt-8 mb-4')
        detail_df = cf.get_annual_cash_flow_table()
        ui.aggrid.from_pandas(detail_df.reset_index()).classes('w-full h-96')

        with ui.expansion("🔗 Cash Flow - Sankey Diagram", icon='account_tree').classes('w-full border rounded mt-4'):
            ui.plotly(plot_cf_sankey_chart(cf, height=600)).classes('w-full')

        # --- Save/Export ---
        ui.separator().classes('my-8')
        ui.label("💾 Save Scenario Result").classes('text-2xl font-bold mb-4')
        with ui.row().classes('w-full items-end'):
            default_name = f"Result_{res['dev_name']}_{res['price_name']}"
            save_name = ui.input("Result Name", value=default_name).classes('flex-grow')
            
            def save_res_to_proj():
                if save_name.value:
                    temp = s.get("cashflow_results", {})
                    temp[save_name.value] = {
                        'cf': cf,
                        'result_summary': cf.get_project_summary(),
                        'inputs': res['inputs']
                    }
                    s["cashflow_results"] = temp
                    save_project(s["current_project"])
                    ui.notify(f"Result '{save_name.value}' saved to project!")
            
            ui.button("Save Result to Project", on_click=save_res_to_proj).classes('ml-2')

        ui.label("📊 Export Results").classes('text-2xl font-bold mt-8 mb-4')
        
        def download_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                detail_df.to_excel(writer, index=False, sheet_name='Annual Cash Flow')
                summ_df = pd.DataFrame(list(summ.items()), columns=['Metric', 'Value'])
                summ_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # NiceGUI download: we need to serve this bytes somehow or use ui.download (if URL)
            # For local app, we can just save to file or use a temporary link.
            # NiceGUI ui.download accepts bytes.
            ui.download(output.getvalue(), f"economic_report_{s['current_project']}_{res['dev_name']}_{res['price_name']}.xlsx")

        ui.button("📥 Download Full Economic Report (Excel)", on_click=download_excel).classes('w-full').props('secondary')

    results_container()
