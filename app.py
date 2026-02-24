import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import base64
import requests
from openai import OpenAI



# --- Constants & Configuration ---
REQUIRED_COLUMNS = [
    'Issue key', 'Summary', 'Status', 'Assignee', 
    'Created', 'Updated', 'Issue Type', 'Priority'
]

STATUS_CATEGORIES = ["TODO", "IN_PROGRESS", "DONE"]

# --- Helper Functions ---

def load_data(uploaded_files):
    """Loads and aggregates data from multiple uploaded CSV/Excel files."""
    all_data = []
    errors = []

    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                except Exception:
                    # Fallback: Try semicolon separator (common in some regions)
                    file.seek(0)
                    try:
                        df = pd.read_csv(file, sep=';')
                    except Exception:
                        # Fallback: Try tab separator
                        file.seek(0)
                        df = pd.read_csv(file, sep='\t', on_bad_lines='skip')
            else:
                df = pd.read_excel(file)
            
            # Check for required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                errors.append(f"Файл '{file.name}' не містить обов'язкових колонок: {', '.join(missing_cols)}")
                continue
            
            # Standardize date columns
            df['Created'] = pd.to_datetime(df['Created'], errors='coerce', utc=True)
            df['Updated'] = pd.to_datetime(df['Updated'], errors='coerce', utc=True)
            
            # Add Source File column for team differentiation
            df['Team_Source'] = file.name
            all_data.append(df)
            
        except Exception as e:
            errors.append(f"Помилка при зчитуванні файлу '{file.name}': {str(e)}")

    if all_data:
        return pd.concat(all_data, ignore_index=True), errors
    return pd.DataFrame(), errors

def calculate_reopens(changelog):
    """Calculates number of reopen events (Done -> To Do/In Progress)."""
    if not changelog:
        return 0
    
    reopen_count = 0
    histories = changelog.get("histories", [])
    
    for history in histories:
        for item in history.get("items", []):
            if item.get("field") == "status":
                from_status = str(item.get("fromString", "")).lower()
                to_status = str(item.get("toString", "")).lower()
                
                # Check for explicit "Reopen" status
                if "reopen" in to_status:
                    reopen_count += 1
                    continue
                
                # Check for Done -> To Do/In Progress transition
                is_from_done = from_status in ["done", "completed", "resolved", "closed"]
                is_to_todo = to_status in ["to do", "open", "in progress", "backlog", "selected for development"]
                
                if is_from_done and is_to_todo:
                    reopen_count += 1
                    
    return reopen_count

def normalize_jira_data(all_issues):
    """Converts raw Jira issues list to a normalized DataFrame."""
    data_list = []
    for issue in all_issues:
        fields = issue.get("fields", {})
        changelog = issue.get("changelog", {})
        
        # Safe parsing
        assignee = fields.get("assignee")
        assignee_name = assignee.get("displayName") if assignee else "Unassigned"
        
        status = fields.get("status")
        status_name = status.get("name") if status else "Unknown"
        
        priority = fields.get("priority")
        priority_name = priority.get("name") if priority else "None"
        
        created_dt = fields.get("created")
        updated_dt = fields.get("updated")
        
        issuetype = fields.get("issuetype")
        issuetype_name = issuetype.get("name") if issuetype else "Unknown"
        
        reopen_count = calculate_reopens(changelog)
        
        # Extract transitions for detailed analysis if needed
        transitions = []
        if changelog:
            for history in changelog.get("histories", []):
                created = history.get("created")
                for item in history.get("items", []):
                    if item.get("field") == "status":
                        transitions.append({
                            "date": created,
                            "from": item.get("fromString"),
                            "to": item.get("toString")
                        })
        
        data_list.append({
            "Issue key": issue.get("key"), # Strict: 'Issue key'
            "Summary": fields.get("summary", ""),
            "Status": status_name,
            "Assignee": assignee_name,
            "Created": pd.to_datetime(created_dt).tz_localize(None) if created_dt else None,
            "Updated": pd.to_datetime(updated_dt).tz_localize(None) if updated_dt else None,
            "Issue Type": issuetype_name,
            "Priority": priority_name,
            "Reopen Count": reopen_count,
            "Team_Source": "Jira API",
            "Transitions": transitions
        })
        
    return pd.DataFrame(data_list)

def fetch_jira_data(jira_url, email, api_token, project_key, days_back):
    """Fetches issues from Jira API and converts to DataFrame with strict columns."""
    # Clean inputs
    project_key = project_key.replace('"', '').replace("'", "").strip()
    jira_url = jira_url.strip()
    
    # Construct JQL
    jql = f"project = '{project_key}' AND updated >= -{days_back}d"
    
    # Prepare API request
    if not jira_url.startswith("http"):
        jira_url = f"https://{jira_url}"
        
    api_endpoint = f"{jira_url.rstrip('/')}/rest/api/3/search/jql"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    all_issues = []
    next_page_token = None
    max_results = 100 
    
    status_placeholder = st.empty()
    fetch_count = 0
    
    try:
        # 1. Fetch Issues
        while True:
            fetch_count += 1
            status_placeholder.text(f"Завантаження сторінки {fetch_count} (всього {len(all_issues)} задач)...")
            
            payload = {
                "jql": jql,
                "maxResults": max_results,
                "fields": ["key", "summary", "status", "assignee", "created", "updated", "issuetype", "priority"],
            }
            
            if next_page_token:
                payload["nextPageToken"] = next_page_token
                
            response = requests.post(
                api_endpoint, 
                json=payload, 
                headers=headers, 
                auth=(email, api_token)
            )
            
            if response.status_code != 200:
                try:
                    error_msg = response.json()
                except:
                    error_msg = response.text
                return None, f"Jira API Error ({response.status_code}): {error_msg} | Payload: {json.dumps(payload)}"
            
            data = response.json()
            batch = data.get("issues", [])
            
            if not batch:
                break
                
            all_issues.extend(batch)
            
            next_page_token = data.get("nextPageToken")
            
            if not next_page_token:
                break
                
        # 2. Fetch Changelog History (Separate Loop for Reliability)
        status_placeholder.text(f"Fetching history for {len(all_issues)} issues...")
        progress_bar = st.progress(0)
        
        for i, issue in enumerate(all_issues):
            issue_key = issue.get("key")
            changelog_url = f"{jira_url.rstrip('/')}/rest/api/3/issue/{issue_key}/changelog"
            
            try:
                cl_response = requests.get(
                    changelog_url,
                    headers=headers,
                    auth=(email, api_token)
                )
                if cl_response.status_code == 200:
                    cl_data = cl_response.json()
                    # Manually attach to issue dict
                    issue["changelog"] = {"histories": cl_data.get("values", [])}
            except Exception:
                pass 
            
            progress_bar.progress((i + 1) / len(all_issues))
            
        status_placeholder.empty()
        progress_bar.empty()
        
        # Normalize Data
        df = normalize_jira_data(all_issues)
        return df, None

    except Exception as e:
        return None, f"Unexpected Error: {str(e)}"

def classify_statuses_with_ai(unique_statuses, api_key):
    """Uses OpenAI to classify statuses into TODO, IN_PROGRESS, DONE."""
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a Jira expert. Map these statuses: {unique_statuses} 
    to exactly one of these three categories: ['TODO', 'IN_PROGRESS', 'DONE'].
    
    Return ONLY a valid JSON object where the key is the status name and the value is the category.
    Do not include any markdown formatting or explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo if preferred for cost
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def calculate_metrics(df, zombie_days_threshold, status_mapping):
    """Calculates all required health metrics using dynamic status mapping."""
    if df.empty:
        return {
            'zombies': pd.DataFrame(),
            'orphans': pd.DataFrame(),
            'dead_epics': pd.DataFrame(),
            'overloaded_assignees': [],
            'wip_counts': pd.Series(),
            'noise_ratio': 0,
            'total_issues': 0,
            'active_issues': 0
        }

    now = pd.Timestamp.now(tz='UTC')
    
    # Map statuses to categories
    # If a status is missing from mapping, default to IN_PROGRESS (fail safe) or keep original
    # We'll Create a 'Category' column
    df['Category'] = df['Status'].map(status_mapping).fillna('IN_PROGRESS')
    
    # Ensure Updated is timezone aware
    if df['Updated'].dt.tz is None:
         df['Updated'] = df['Updated'].dt.tz_localize('UTC')

    # 1. Zombie Tasks: Category is TODO or IN_PROGRESS, no updates > X days
    zombie_mask = (
        (df['Category'].isin(['TODO', 'IN_PROGRESS'])) & 
        (df['Updated'] < (now - pd.Timedelta(days=zombie_days_threshold)))
    )
    zombies = df[zombie_mask].copy()
    
    # 2. Orphan Tasks: Unassigned
    orphan_mask = df['Assignee'].isna() | (df['Assignee'] == '') | (df['Assignee'].astype(str).str.strip() == 'nan')
    orphans = df[orphan_mask].copy()
    
    # 3. Dead Epics: Type=Epic, Category != DONE, No updates > 30 days
    epic_mask = (
        (df['Issue Type'] == 'Epic') & 
        (df['Category'] != 'DONE') & 
        (df['Updated'] < (now - pd.Timedelta(days=30)))
    )
    dead_epics = df[epic_mask].copy()
    
    # 4. WIP Overload: Assignees with > 2 active (TODO/IN_PROGRESS) tasks
    active_tasks = df[df['Category'] != 'DONE']
    wip_counts = active_tasks['Assignee'].value_counts()
    overloaded_assignees = wip_counts[wip_counts > 2].index.tolist()
    
    # 5. Noise Ratio: Reopened / DONE (Advanced Logic)
    noisy_issues_count = 0
    
    if 'Transitions' in df.columns:
        # API Mode: Use Changelog
        for _, row in df.iterrows():
            transitions = row['Transitions']
            is_noisy = False
            
            if isinstance(transitions, list):
                for t in transitions:
                    # Logic 1: Transition contains "Reopen" word
                    to_status = str(t.get('to', '')).lower()
                    if "reopen" in to_status:
                        is_noisy = True
                        break
                    
                    # Logic 2: Backwards transition (DONE -> TODO/IN_PROGRESS)
                    from_cat = status_mapping.get(t.get('from'), 'IN_PROGRESS')
                    to_cat = status_mapping.get(t.get('to'), 'IN_PROGRESS')
                    
                    if from_cat == 'DONE' and to_cat in ['TODO', 'IN_PROGRESS']:
                        is_noisy = True
                        break
            
            if is_noisy:
                noisy_issues_count += 1
    else:
        # CSV Mode: Fallback to current status check
        reopened_mask = df['Status'].str.contains('reopen', case=False, regex=False) | df['Status'].str.contains('open', case=False, regex=False)
        # Proper fallback: check for "Reopen" in status name only
        noisy_issues_count = df[df['Status'].str.lower().str.contains('reopen')].shape[0] if not df.empty else 0
    
    done_count = df[df['Category'] == 'DONE'].shape[0]
    noise_ratio = (noisy_issues_count / done_count * 100) if done_count > 0 else 0
    
    return {
        'zombies': zombies,
        'orphans': orphans,
        'dead_epics': dead_epics,
        'overloaded_assignees': overloaded_assignees,
        'wip_counts': wip_counts,
        'noise_ratio': noise_ratio,
        'total_issues': len(df),
        'active_issues': len(active_tasks)
    }

# --- Main UI ---

def main():
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Аудит здоров'я беклогу Jira (AI Powered)",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Аудит здоров'я беклогу Jira (AI)")
    
    # --- Sidebar ---
    st.sidebar.header("Налаштування")
    
    openai_api_key = st.sidebar.text_input("OpenAI API Key", value="", type="password")
    if not openai_api_key:
        st.sidebar.warning("Введіть API Key для авто-класифікації статусів!")
    
    # Data Source Selection
    data_source = st.sidebar.radio("Джерело даних", ["CSV Upload", "Jira API"])
    
    zombie_threshold = st.sidebar.slider(
        "Поріг 'Зомбі' задач (днів)", 
        min_value=7, 
        max_value=90, 
        value=14
    )

    
    full_df = pd.DataFrame()
    errors = []
    
    if data_source == "CSV Upload":
        uploaded_files = st.sidebar.file_uploader(
            "Завантажте експорти Jira (CSV/Excel)", 
            type=['csv', 'xlsx'], 
            accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("Будь ласка, завантажте файли для початку аналізу.")
            # Show Documentation Tab even if no files
            show_documentation_tab()
            return

        # --- Data Processing ---
        full_df, errors = load_data(uploaded_files)

    else: # Jira API
        st.sidebar.subheader("Підключення до Jira")
        jira_url = st.sidebar.text_input("Jira Server URL", placeholder="https://company.atlassian.net")
        jira_email = st.sidebar.text_input("Email / Username")
        jira_token = st.sidebar.text_input("API Token", type="password", help="Згенеруйте токен в налаштуваннях профілю Atlassian")
        
        project_key = st.sidebar.text_input("Project Key (напр. PROJ)")
        timeframe_days = st.sidebar.selectbox("Період аналізу", [30, 90, 180], index=0) # Default 30 days
        
        # Load from cache if available
        if 'jira_data_cache' in st.session_state:
            full_df = st.session_state['jira_data_cache']
            st.sidebar.info(f"Використовуються дані з кешу ({len(full_df)} задач).")

        if st.sidebar.button("Завантажити дані"):
            with st.spinner("З'єднання з Jira..."):
                api_df, error = fetch_jira_data(jira_url, jira_email, jira_token, project_key, timeframe_days)
                
                if error:
                    st.error(error)
                    show_documentation_tab()
                    return
                
                full_df = api_df
                st.session_state['jira_data_cache'] = full_df
                st.sidebar.success(f"Завантажено {len(full_df)} задач!")

        if full_df.empty:
            st.info("Введіть дані для підключення та натисніть 'Завантажити дані'.")
            show_documentation_tab()
            return
    
    if errors:
        for err in errors:
            st.error(err)
        if full_df.empty:
            return

    # --- Status Management ---
    unique_statuses = sorted(full_df['Status'].astype(str).unique().tolist())
    
    # Initialize session state for mapping
    if 'status_mapping' not in st.session_state:
        st.session_state['status_mapping'] = {}
    
    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Налаштування статусів", 
        "Загальний огляд", 
        "Візуальна аналітика", 
        "Задачі до розгляду",
        "Документація (Help)"
    ])

    # --- Tab 1: Status Settings (Mapping) ---
    with tab1:
        st.header("Налаштування статусів (Status Mapping)")
        st.markdown("Зіставте статуси Jira з категоріями: **TODO**, **IN_PROGRESS**, **DONE**.")
        
        col_btn, col_info = st.columns([1, 4])
        
        with col_btn:
            if st.button("🪄 Авто-класифікація (AI)"):
                if not openai_api_key:
                    st.error("Потрібен OpenAI API Key!")
                else:
                    with st.spinner("AI аналізує статуси..."):
                        ai_mapping = classify_statuses_with_ai(unique_statuses, openai_api_key)
                        if ai_mapping:
                            st.session_state['status_mapping'].update(ai_mapping)
                            st.success("Класифікацію завершено!")
        
        # Prepare Data Editor
        mapping_data = []
        for status in unique_statuses:
            current_cat = st.session_state['status_mapping'].get(status, "TODO") # Default to TODO
            mapping_data.append({"Original Status": status, "Category": current_cat})
        
        mapping_df = pd.DataFrame(mapping_data)
        
        edited_df = st.data_editor(
            mapping_df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    help="Оберіть категорію для статусу",
                    width="medium",
                    options=STATUS_CATEGORIES,
                    required=True,
                )
            },
            hide_index=True,
            use_container_width=True,
            key="status_editor"
        )
        
        # Update session state from editor
        final_mapping = dict(zip(edited_df["Original Status"], edited_df["Category"]))
        st.session_state['status_mapping'] = final_mapping

    # --- Apply Mapping & Calculate Metrics ---
    # Filter functionality
    team_options = ["Всі команди (Портфоліо)"] + list(full_df['Team_Source'].unique())
    selected_view = st.sidebar.selectbox("Оберіть рівень аналізу", team_options)

    if selected_view != "Всі команди (Портфоліо)":
        analysis_df = full_df[full_df['Team_Source'] == selected_view].copy()
    else:
        analysis_df = full_df.copy()

    metrics = calculate_metrics(analysis_df, zombie_threshold, st.session_state['status_mapping'])

    # --- Tab 2: Executive Summary ---
    with tab2:
        st.header(f"Звіт: {selected_view}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всього задач", metrics['total_issues'])
        col2.metric("Активні задачі", metrics['active_issues'])
        col3.metric("Зомбі задачі", len(metrics['zombies']), delta_color="inverse")
        col4.metric("Без виконавця", len(metrics['orphans']), delta_color="inverse")
        
        st.divider()
        
        # Health Score
        penalty = (len(metrics['zombies']) * 2) + (len(metrics['orphans']) * 1) + (len(metrics['dead_epics']) * 5)
        health_score = max(0, 100 - (penalty / max(1, metrics['active_issues']) * 10))
        
        st.subheader("Оцінка здоров'я потоку")
        st.progress(health_score / 100)
        st.caption(f"Поточний бал: {health_score:.1f}/100")

        if health_score < 50:
            st.error("🚨 Критичний стан! Рекомендовано терміновий грумінг беклогу.")
        elif health_score < 80:
            st.warning("⚠️ Потрібна увага. Є застарілі задачі та прогалини у розподілі.")
        else:
            st.success("✅ Беклог у гарному стані. Продовжуйте підтримувати гігієну.")

        st.markdown(f"""
        **Інсайти:**
        - **Коефіцієнт шуму (Reopened/Done):** {metrics['noise_ratio']:.1f}%
        - **"Мертві" Епіки:** {len(metrics['dead_epics'])}
        - **Перевантажені виконавці:** {len(metrics['overloaded_assignees'])}
        """)

    # --- Tab 3: Visual Analytics ---
    with tab3:
        st.subheader("Розподіл за категоріями (Mapping)")
        cat_counts = analysis_df['Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_pie = px.pie(
            cat_counts, 
            values='Count', 
            names='Category', 
            title='Структура беклогу (TODO/IN_PROGRESS/DONE)', 
            hole=0.4,
            color='Category',
            color_discrete_map={'TODO':'#ebecf0', 'IN_PROGRESS':'#0052cc', 'DONE':'#36b37e'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("Навантаження на команду")
        if not metrics['wip_counts'].empty:
            wip_df = metrics['wip_counts'].sort_values(ascending=True).tail(15)
            fig_bar = px.bar(
                x=wip_df.values, 
                y=wip_df.index, 
                orientation='h',
                title="Топ виконавців за кількістю активних задач",
                labels={'x': 'Кількість задач', 'y': 'Виконавець'},
            )
            fig_bar.add_vline(x=2, line_dash="dash", line_color="red", annotation_text="Ліміт (2)")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- Tab 4: Actionable Backlog ---
    with tab4:
        st.subheader("🔍 Детальний список проблемних задач")
        
        problem_type = st.selectbox(
            "Оберіть категорію:",
            ["Зомбі задачі (Zombie)", "Задачі без виконавця (Orphan)", "Мертві епіки (Dead Epics)"]
        )
        
        if problem_type == "Зомбі задачі (Zombie)":
            target_df = metrics['zombies']
        elif problem_type == "Задачі без виконавця (Orphan)":
            target_df = metrics['orphans']
        else:
            target_df = metrics['dead_epics']

        if not target_df.empty:
            st.dataframe(
                target_df[['Issue key', 'Summary', 'Assignee', 'Status', 'Category', 'Updated', 'Team_Source']].sort_values(by='Updated'),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("Чудово! У цій категорії не знайдено проблемних задач.")

    # --- Tab 5: Documentation ---
    with tab5:
        show_documentation_content()

def show_documentation_tab():
    """Helper to show documentation when no data is loaded."""
    st.divider()
    st.header("📚 Документація")
    show_documentation_content()

def show_documentation_content():
    try:
        with open("DOCUMENTATION.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("Файл документації (DOCUMENTATION.md) не знайдено.")

if __name__ == "__main__":
    main()
