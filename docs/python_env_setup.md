## Setting Up the Python Environment and Installing Dependencies

This guide explains how to set up a Python environment using `venv` and install the required dependencies from the `requirements.txt` file.

### Prerequisites
- Python 3.11 must be installed on your system.
- Ensure `pip` is installed and accessible.

### Steps to Set Up the Environment

1. **Create a Virtual Environment:**
   Open a terminal in the root directory of the project and run the following command to create a virtual environment:
   ```bash
   python -m venv name_of_virtual_environment
   ```
   Replace `name_of_virtual_environment` with your desired name for the environment.

2. **Activate the Virtual Environment:**
   - On **Windows**:
     ```bash
     name_of_virtual_environment\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source name_of_virtual_environment/bin/activate
     ```

3. **Upgrade `pip`:**
   It is recommended to upgrade `pip` to the latest version before installing dependencies:
   ```bash
   pip install --upgrade pip
   ```

4. **Install Dependencies:**
   Use the following command to install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation:**
   Check if the necessary packages are installed correctly by running:
   ```bash
   pip list
   ```
   This will display all installed packages and their versions.

6. **Deactivate the Virtual Environment:**
   Once you have completed your work, deactivate the virtual environment by running:
   ```bash
   deactivate
   ```

### Notes
- Always activate the virtual environment before working on the project to ensure you use the correct Python environment.
- If new dependencies are added to the project, update the `requirements.txt` file using:
  ```bash
  pip freeze > requirements.txt
  ```
  This will regenerate the `requirements.txt` file with all currently installed packages.

By following these steps, you can ensure a consistent and isolated environment for your project.

