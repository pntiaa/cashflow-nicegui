from nicegui import ui, app
from utils import ensure_state_init, get_all_projects_summary
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

# --- Initialization ---
# storage_secret is required for app.storage.user
app.add_static_files('/static', 'static') # Placeholder if needed

@ui.page('/')
def index_page():
    ensure_state_init()
    with layout("Summary"):
        ui.label("💰 Cashflow Analysis App").classes('text-3xl font-bold mb-4')
        
        ui.markdown("""
        Welcome to the **Cashflow Analysis App**. This tool allows you to perform end-to-end economic evaluations of oil and gas projects.
        
        ### Workflow:
        1.  **Production**: Define your type curves and generate field production profiles.
        2.  **Development**: Create development cost scenarios based on your drilling plans.
        3.  **Price Deck**: Set up your oil and gas price forecasts and inflation expectations.
        4.  **Cash Flow**: Combine your saved cases to calculate NPV, IRR, and overall project economics.
        """)
        
        ui.separator().classes('my-6')
        
        ui.label("📊 Project Portfolio Summary").classes('text-2xl font-semibold mb-2')
        summary_df = get_all_projects_summary()
        if not summary_df.empty:
             ui.aggrid.from_pandas(summary_df).classes('w-full h-64')
        else:
             ui.label("No projects found. Create one in the sidebar!")

# --- Shared Layout ---
def layout(title: str):
    ui.query('body').style('background-color: #f8f9fa')
    
    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        with ui.row().classes('items-center'):
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
            ui.label(f'Cashflow Analysis - {title}').classes('font-bold text-lg')
        
        with ui.row().classes('items-center'):
            if app.storage.user.get('current_project'):
                ui.badge(f"Project: {app.storage.user['current_project']}", color='green')
            else:
                ui.badge("No active project", color='orange')

    with ui.left_drawer(value=True).classes('bg-slate-100') as left_drawer:
        with ui.column().classes('w-full p-4'):
             ui.label("Navigation").classes('text-xs font-bold text-slate-500 uppercase mb-2')
             ui.link('Dashboard', '/').classes('text-blue-600 hover:underline mb-1')
             ui.link('Settings', '/settings').classes('text-blue-600 hover:underline mb-1')
             ui.link('Development', '/development').classes('text-blue-600 hover:underline mb-1')
             ui.link('Price Deck', '/price_deck').classes('text-blue-600 hover:underline mb-1')
             ui.link('Cash Flow', '/cash_flow').classes('text-blue-600 hover:underline mb-1')
             
             ui.separator().classes('my-4')
             
             # Project management part
             render_project_sidebar_ui()

    return ui.column().classes('max-w-5xl mx-auto p-8 bg-white border rounded-lg shadow-sm mt-4 w-full')

# Placeholder for project sidebar UI
def render_project_sidebar_ui():
    from utils import list_projects, load_project, save_project
    
    ui.label("📁 Project Management").classes('text-lg font-bold mb-2')
    
    with ui.tabs().classes('w-full') as tabs:
        select_tab = ui.tab('Select')
        create_tab = ui.tab('Create')
        
    with ui.tab_panels(tabs, value=select_tab).classes('w-full bg-transparent'):
        with ui.tab_panel(select_tab):
            projects = list_projects()
            if projects:
                selected = ui.select(projects, label='Select Project', value=app.storage.user.get('current_project')).classes('w-full')
                ui.button('Load Project', on_click=lambda: [load_project(selected.value), ui.notify(f"Loaded {selected.value}"), ui.navigate.to('/')]).classes('w-full mt-2')
            else:
                ui.label("No projects found.")
                
        with ui.tab_panel(create_tab):
            name = ui.input('New Project Name').classes('w-full')
            def do_create():
                if name.value:
                    app.storage.user['current_project'] = name.value
                    app.storage.user['production_cases'] = {}
                    app.storage.user['development_cases'] = {}
                    app.storage.user['price_cases'] = {}
                    app.storage.user['cashflow_results'] = {}
                    save_project(name.value)
                    ui.notify(f"Project {name.value} created!")
                    ui.navigate.to('/')
                else:
                    ui.notify("Please enter a name", type='negative')
            ui.button('Create Project', on_click=do_create).classes('w-full mt-2')

from ui import settings, development, price_deck, cash_flow

# Define other pages as placeholders
@ui.page('/settings')
def settings_page():
    ensure_state_init()
    with layout("Settings"):
        settings.content()

@ui.page('/development')
def development_page():
    ensure_state_init()
    with layout("Development"):
        development.content()

@ui.page('/price_deck')
def price_deck_page():
    ensure_state_init()
    with layout("Price Deck"):
        price_deck.content()

@ui.page('/cash_flow')
def cash_flow_page():
    ensure_state_init()
    with layout("Cash Flow"):
        cash_flow.content()

if __name__ in {"__main__", "__mp_main__"}:
# if __name__ in {"__main__", "reloader"}:
    # Use a secret for storage
    ui.run(storage_secret='cashflow_secret_key_123')
