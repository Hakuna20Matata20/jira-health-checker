import pandas as pd
import random
from datetime import datetime, timedelta

def generate_jira_data():
    # Define columns
    columns = ['Issue Key', 'Summary', 'Status', 'Assignee', 'Created', 'Updated', 'Issue Type']
    
    # Initialize list for data
    data = []
    
    # Common values
    developers = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
    statuses_en = ['To Do', 'In Progress', 'Done', 'Review', 'Blocked']
    statuses_ru_ua = ['В работе', 'Готово', 'Ожидает', 'Тестирование', 'Новая']
    issue_types = ['Task', 'Bug', 'Story', 'Epic']
    
    # Helper to generate random date within last 60 days
    def random_date(start_days_ago=60, end_days_ago=0):
        start_date = datetime.now() - timedelta(days=start_days_ago)
        end_date = datetime.now() - timedelta(days=end_days_ago)
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        return start_date + timedelta(seconds=random_seconds)

    # 1. Generate 'Overloaded User' tasks (20 tasks for one specific user)
    # Let's say 'Frank' is the overloaded user.
    overloaded_user = 'Frank'
    # The requirement says "5 - 'Overloaded users' (One person having 20 tasks)".
    # This might mean 5 *people* are overloaded, or one person has 20 tasks.
    # Re-reading: "5 - 'Overloaded users' (One person having 20 tasks)."
    # This is slightly ambiguous. "5 - 'Overloaded users'" usually implies a category count in a list of requirements.
    # But the parenthetical says "(One person having 20 tasks)".
    # If I interpret "5 - 'Overloaded users'" as a bullet point number 5, it means "Create a scenario with an overloaded user".
    # However, looking at the previous items:
    # "10 - 'Zombie tasks'..." (10 tasks)
    # "5 - 'Orphan tasks'..." (5 tasks)
    # "5 - 'Overloaded users' (One person having 20 tasks)."
    # This likely means "Create 5 tasks that contribute to an overloaded user scenario" OR "Create an overloaded user scenario, and maybe the number 5 is a typo or refers to something else?"
    # Actually, if one person has 20 tasks, that takes up 20 rows.
    # If the requirement is just "One person having 20 tasks", I will dedicate 20 rows to 'Frank'.
    # But the prompt says "5 - 'Overloaded users'".
    # Let's check the formatting:
    # 10 - 'Zombie tasks'
    # 5 - 'Orphan tasks'
    # 5 - 'Overloaded users' (One person having 20 tasks)
    # 10 - tasks with Russian...
    # It looks like a list of counts.
    # Maybe it means "5 users who are overloaded"? No, "(One person having 20 tasks)" suggests a single case.
    # Maybe it means "Create 5 tasks assign to the overloaded user, and ensuring they have 20 total?"
    # I will assume the user wants one specific user to have 20 tasks total in the dataset.
    # The "5" might be a mistake or refers to something else, but "One person having 20 tasks" is very specific.
    # Use 'Frank' and give him 20 tasks.
    # Wait, if I make 20 tasks for Frank, that consumes 20 rows.
    # I'll create 20 tasks for Frank.
    
    for _ in range(20):
        data.append({
            'Issue Key': f'JIRA-{len(data)+1}',
            'Summary': f'Task for overloaded user Frank',
            'Status': random.choice(statuses_en),
            'Assignee': overloaded_user,
            'Created': random_date(60, 30),
            'Updated': random_date(30, 0),
            'Issue Type': random.choice(issue_types)
        })

    # 2. Generate 10 'Zombie tasks' (Updated > 30 days ago)
    for _ in range(10):
        data.append({
            'Issue Key': f'JIRA-{len(data)+1}',
            'Summary': 'Zombie task - old update',
            'Status': random.choice(statuses_en),
            'Assignee': random.choice(developers),
            'Created': random_date(90, 61),
            'Updated': random_date(60, 31), # More than 30 days ago
            'Issue Type': random.choice(issue_types)
        })

    # 3. Generate 5 'Orphan tasks' (Assignee is empty)
    for _ in range(5):
        data.append({
            'Issue Key': f'JIRA-{len(data)+1}',
            'Summary': 'Orphan task - no assignee',
            'Status': random.choice(statuses_en),
            'Assignee': '', # Empty
            'Created': random_date(30, 0),
            'Updated': random_date(10, 0),
            'Issue Type': random.choice(issue_types)
        })

    # 4. Generate 10 tasks with Russian or Ukrainian statuses
    for _ in range(10):
        data.append({
            'Issue Key': f'JIRA-{len(data)+1}',
            'Summary': 'Task with non-English status',
            'Status': random.choice(statuses_ru_ua),
            'Assignee': random.choice(developers),
            'Created': random_date(30, 0),
            'Updated': random_date(10, 0),
            'Issue Type': random.choice(issue_types)
        })
        
    # 5. Fill the rest to reach 100 rows
    remaining_count = 100 - len(data)
    for _ in range(remaining_count):
        data.append({
            'Issue Key': f'JIRA-{len(data)+1}',
            'Summary': 'Standard task',
            'Status': random.choice(statuses_en),
            'Assignee': random.choice(developers),
            'Created': random_date(30, 0),
            'Updated': random_date(10, 0),
            'Issue Type': random.choice(issue_types)
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'Jira_test_data.csv'
    df.to_csv(output_file, index=False)
    print(f"File '{output_file}' generated with strict requirements.")
    print(f"Total rows: {len(df)}")
    print(f"Zombie tasks (>30 days): {len(df[df['Updated'] < (datetime.now() - timedelta(days=30))])}")
    print(f"Orphan tasks (No Assignee): {len(df[df['Assignee'] == ''])}")
    print(f"Overloaded user ({overloaded_user}) tasks: {len(df[df['Assignee'] == overloaded_user])}")
    
    # Check for Russian/Ukrainian characters in Status
    ru_ua_count = df['Status'].apply(lambda x: any(ord(c) > 127 for c in str(x))).sum()
    print(f"Tasks with potential non-ASCII statuses: {ru_ua_count}")

if __name__ == "__main__":
    generate_jira_data()
