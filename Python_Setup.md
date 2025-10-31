# 🛠️ Python Environment Setup for DSA

## 📋 Environment Information
- **Python Version**: 3.12.11
- **Environment Type**: Standard Python Installation
- **Python Executable**: `C:/msys64/ucrt64/bin/python.exe`

## 📦 Required Packages

```bash
# Install essential packages for DSA practice
pip install numpy matplotlib jupyter notebook pytest
```

### Package Purposes:
- **numpy**: For mathematical operations and array handling
- **matplotlib**: For visualizing algorithms and data structures  
- **jupyter**: Interactive coding and documentation
- **pytest**: For testing your solutions

## 🚀 Quick Setup Commands

### 1. Install Packages
```powershell
C:/msys64/ucrt64/bin/python.exe -m pip install numpy matplotlib jupyter notebook pytest
```

### 2. Run Python Scripts
```powershell
# Instead of: python script.py  
C:/msys64/ucrt64/bin/python.exe script.py
```

### 3. Start Jupyter Notebook (Optional)
```powershell
C:/msys64/ucrt64/bin/python.exe -m jupyter notebook
```

### 4. Test Installation
```powershell
C:/msys64/ucrt64/bin/python.exe -c "import numpy, matplotlib; print('All packages installed successfully!')"
```

## 💡 Development Tips

1. **Use VS Code**: Best IDE for Python development
2. **Code Templates**: Use the templates in `/01_Python_Basics/` folder  
3. **Testing**: Always test your solutions with different inputs
4. **Documentation**: Comment your code in Hindi/English for better understanding

## 🔧 Troubleshooting

### If pip install fails:
```powershell
C:/msys64/ucrt64/bin/python.exe -m pip install --upgrade pip
C:/msys64/ucrt64/bin/python.exe -m pip install --user package_name
```

### Check Python version:
```powershell
C:/msys64/ucrt64/bin/python.exe --version
```

### List installed packages:
```powershell
C:/msys64/ucrt64/bin/python.exe -m pip list
```