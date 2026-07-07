# Employee Management System (EMS)

# Step 1 - Data Storage
employees = {
    101: {'name': 'Satya', 'age': 27, 'department': 'HR', 'salary': 50000},
    102: {'name': 'Riya', 'age': 30, 'department': 'Engineering', 'salary': 75000},
    103: {'name': 'Arjun', 'age': 25, 'department': 'Marketing', 'salary': 45000},
}


# Step 3 - Add Employee
def add_employee():
    print("\n--- Add Employee ---")

    while True:
        try:
            emp_id = int(input("Enter Employee ID: "))
        except ValueError:
            print("Invalid input. Employee ID must be a number.")
            continue

        if emp_id in employees:
            print(f"Employee ID {emp_id} already exists. Please enter a unique ID.")
        else:
            break

    name = input("Enter Employee Name: ").strip()

    while True:
        try:
            age = int(input("Enter Employee Age: "))
            break
        except ValueError:
            print("Invalid input. Age must be a number.")

    department = input("Enter Employee Department: ").strip()

    while True:
        try:
            salary = float(input("Enter Employee Salary: "))
            break
        except ValueError:
            print("Invalid input. Salary must be a number.")

    employees[emp_id] = {
        'name': name,
        'age': age,
        'department': department,
        'salary': salary
    }

    print(f"\nEmployee '{name}' added successfully!")


# Step 4 - View All Employees
def view_employees():
    print("\n--- All Employees ---")

    if not employees:
        print("No employees available.")
        return

    print(f"{'ID':<10} {'Name':<20} {'Age':<8} {'Department':<20} {'Salary':<10}")
    print("-" * 70)

    for emp_id, details in employees.items():
        print(f"{emp_id:<10} {details['name']:<20} {details['age']:<8} {details['department']:<20} {details['salary']:<10}")


# Step 5 - Search Employee
def search_employee():
    print("\n--- Search Employee ---")

    try:
        emp_id = int(input("Enter Employee ID to search: "))
    except ValueError:
        print("Invalid input. Employee ID must be a number.")
        return

    if emp_id in employees:
        details = employees[emp_id]
        print(f"\nEmployee Found:")
        print(f"  ID         : {emp_id}")
        print(f"  Name       : {details['name']}")
        print(f"  Age        : {details['age']}")
        print(f"  Department : {details['department']}")
        print(f"  Salary     : {details['salary']}")
    else:
        print("Employee not found.")


# Step 2 - Menu System
def main_menu():
    while True:
        print("\n========== Employee Management System ==========")
        print("1. Add Employee")
        print("2. View All Employees")
        print("3. Search for Employee")
        print("4. Exit")
        print("================================================")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            add_employee()
        elif choice == '2':
            view_employees()
        elif choice == '3':
            search_employee()
        elif choice == '4':
            print("\nThank you for using EMS. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


# Entry Point
if __name__ == "__main__":
    main_menu()