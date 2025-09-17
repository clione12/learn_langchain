import os
from dotenv import load_dotenv 
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import matplotlib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pymysql
from langchain_tavily import TavilySearch
import requests
from bs4 import BeautifulSoup
import subprocess
import platform
import psutil
import datetime
from urllib.parse import quote
import time
import csv
from pathlib import Path



load_dotenv(override=True)


# ✅ 创建提示词模板
prompt = """
你是一名经验丰富的智能助手，擅长帮助用户高效完成各种任务。你拥有以下强大的工具能力：

## 📁 文件和系统操作工具
1. **文件操作 (file_operations)**: 读取、写入、追加、列出目录、删除文件、检查文件存在性
2. **系统信息 (get_system_info)**: 获取操作系统、CPU、内存、磁盘等系统信息
3. **命令执行 (execute_command)**: 安全地执行系统命令（禁止危险命令）
4. **时间日期 (datetime_operations)**: 获取当前时间、格式化日期、日期计算等
5. **CSV处理 (csv_operations)**: 读取、写入、分析CSV文件

## 💾 数据库和数据分析工具
6. **Python执行 (python_inter)**: 执行非绘图类Python代码，数据处理和计算
7. **绘图工具 (fig_inter)**: 执行matplotlib/seaborn绘图代码并保存图片


## 🎯 工具使用指南

**文件操作时：**
- 读写文件 → 使用 `file_operations`
- 处理CSV数据 → 使用 `csv_operations`
- 系统信息查询 → 使用 `get_system_info`
- 执行命令 → 使用 `execute_command`（安全限制）

**数据分析时：**
- 数据处理 → 使用 `python_inter`
- 数据可视化 → 使用 `fig_inter`（必须创建fig对象，不要用plt.show()）

**时间处理时：**
- 获取当前时间、日期计算 → 使用 `datetime_operations`

## ⚠️ 重要注意事项

1. **绘图要求：**
   - 必须使用 `fig = plt.figure()` 或 `fig, ax = plt.subplots()` 创建图像对象
   - 不要使用 `plt.show()`
   - 图表标签和标题使用英文
   - 调用 `fig.tight_layout()`

2. **安全限制：**
   - 命令执行工具禁止危险命令（如rm -rf, format等）
   - 文件操作限制在安全范围内

3. **输出格式：**
   - 使用**简体中文**回答
   - JSON数据要提取关键信息简要说明
   - 生成图片时使用Markdown格式：`![描述](images/图片名.png)`
   - 保持专业、简洁的风格

4. **工具选择优先级：**
   - 文件处理 → 专用文件工具 → 通用Python工具

请根据用户需求选择最合适的工具，提供准确、高效的帮助。
"""

# ✅ 创建文件操作工具
class FileOperationSchema(BaseModel):
    operation: str = Field(description="操作类型: read, write, append, list, delete, exists")
    file_path: str = Field(description="文件路径")
    content: str = Field(default="", description="写入的内容（仅用于write和append操作）")

@tool(args_schema=FileOperationSchema)
def file_operations(operation: str, file_path: str, content: str = "") -> str:
    """
    执行文件操作，包括读取、写入、追加、列出目录、删除文件、检查文件是否存在。
    支持的操作类型：read, write, append, list, delete, exists
    """
    try:
        path = Path(file_path)
        
        if operation == "read":
            if path.exists() and path.is_file():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 2000:
                        content = content[:2000] + "\n\n[文件内容已截断...]"
                    return content
            else:
                return f"文件不存在: {file_path}"
                
        elif operation == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ 文件已写入: {file_path}"
            
        elif operation == "append":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"✅ 内容已追加到文件: {file_path}"
            
        elif operation == "list":
            if path.exists() and path.is_dir():
                items = []
                for item in path.iterdir():
                    item_type = "目录" if item.is_dir() else "文件"
                    size = item.stat().st_size if item.is_file() else "-"
                    items.append(f"{item_type}: {item.name} ({size} bytes)")
                return "\n".join(items[:50])  # 限制显示50个项目
            else:
                return f"目录不存在: {file_path}"
                
        elif operation == "delete":
            if path.exists():
                if path.is_file():
                    path.unlink()
                    return f"✅ 文件已删除: {file_path}"
                else:
                    return f"❌ 无法删除目录，请使用专门的目录删除工具"
            else:
                return f"文件不存在: {file_path}"
                
        elif operation == "exists":
            return f"文件存在: {path.exists()}"
            
        else:
            return f"不支持的操作类型: {operation}"
            
    except Exception as e:
        return f"文件操作失败: {str(e)}"


# ✅ 创建系统信息工具
@tool
def get_system_info() -> str:
    """
    获取当前系统的基本信息，包括操作系统、CPU、内存、磁盘使用情况等。
    """
    try:
        info = {
            "操作系统": platform.system(),
            "系统版本": platform.release(),
            "处理器": platform.processor(),
            "CPU核心数": psutil.cpu_count(logical=False),
            "逻辑CPU数": psutil.cpu_count(logical=True),
            "CPU使用率": f"{psutil.cpu_percent(interval=1):.1f}%",
            "内存总量": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "内存使用率": f"{psutil.virtual_memory().percent:.1f}%",
            "磁盘使用率": f"{psutil.disk_usage('/').percent:.1f}%",
            "当前时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(info, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"获取系统信息失败: {str(e)}"
    
# ✅ 创建命令执行工具
class CommandSchema(BaseModel):
    command: str = Field(description="要执行的命令")
    timeout: int = Field(default=30, description="超时时间（秒）")


@tool(args_schema=CommandSchema)
def execute_command(command: str, timeout: int = 30) -> str:
    """
    执行系统命令并返回结果。
    注意：仅执行安全的命令，避免执行可能损害系统的命令。
    """
    try:
        # 安全检查：禁止执行危险命令
        dangerous_commands = ['rm -rf', 'del', 'format', 'fdisk', 'mkfs', 'dd if=', 'shutdown', 'reboot']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return "❌ 拒绝执行潜在危险命令"
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout if result.stdout else result.stderr
        if len(output) > 2000:
            output = output[:2000] + "\n\n[输出已截断...]"
            
        return f"命令执行结果 (返回码: {result.returncode}):\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"❌ 命令执行超时 ({timeout}秒)"
    except Exception as e:
        return f"❌ 命令执行失败: {str(e)}"

# ✅ 创建时间和日期工具
class DateTimeSchema(BaseModel):
    operation: str = Field(description="操作类型: now, format, calculate, timezone")
    date_string: str = Field(default="", description="日期字符串（用于格式化或计算）")
    format_string: str = Field(default="%Y-%m-%d %H:%M:%S", description="日期格式")
    days_offset: int = Field(default=0, description="天数偏移量（用于日期计算）")

@tool(args_schema=DateTimeSchema)
def datetime_operations(operation: str, date_string: str = "", format_string: str = "%Y-%m-%d %H:%M:%S", days_offset: int = 0) -> str:
    """
    执行日期时间相关操作，包括获取当前时间、格式化日期、日期计算等。
    """
    try:
        if operation == "now":
            return datetime.datetime.now().strftime(format_string)
            
        elif operation == "format":
            if date_string:
                # 尝试解析日期字符串
                dt = datetime.datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                return dt.strftime(format_string)
            else:
                return "请提供要格式化的日期字符串"
                
        elif operation == "calculate":
            base_date = datetime.datetime.now()
            if date_string:
                base_date = datetime.datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            
            new_date = base_date + datetime.timedelta(days=days_offset)
            return new_date.strftime(format_string)
            
        elif operation == "timezone":
            import time
            return f"当前时区: {time.tzname[0]}, UTC偏移: {time.timezone // 3600} 小时"
            
        else:
            return f"不支持的操作类型: {operation}"
            
    except Exception as e:
        return f"日期时间操作失败: {str(e)}"

# ✅ 创建CSV数据处理工具
class CSVOperationSchema(BaseModel):
    operation: str = Field(description="操作类型: read, write, analyze")
    file_path: str = Field(description="CSV文件路径")
    data: str = Field(default="", description="CSV数据（JSON格式，用于写入操作）")

@tool(args_schema=CSVOperationSchema)
def csv_operations(operation: str, file_path: str, data: str = "") -> str:
    """
    执行CSV文件操作，包括读取、写入、基本分析。
    """
    try:
        if operation == "read":
            df = pd.read_csv(file_path)
            # 限制显示行数
            if len(df) > 10:
                preview = df.head(10)
                return f"CSV文件预览（前10行）：\n{preview.to_string()}\n\n总行数: {len(df)}, 总列数: {len(df.columns)}"
            else:
                return f"CSV文件内容：\n{df.to_string()}"
                
        elif operation == "write":
            if data:
                import json
                json_data = json.loads(data)
                df = pd.DataFrame(json_data)
                df.to_csv(file_path, index=False)
                return f"✅ 数据已写入CSV文件: {file_path}"
            else:
                return "请提供要写入的数据（JSON格式）"
                
        elif operation == "analyze":
            df = pd.read_csv(file_path)
            analysis = {
                "行数": len(df),
                "列数": len(df.columns),
                "列名": list(df.columns),
                "数据类型": df.dtypes.to_dict(),
                "缺失值": df.isnull().sum().to_dict(),
                "数值列统计": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else "无数值列"
            }
            return json.dumps(analysis, ensure_ascii=False, indent=2, default=str)
            
        else:
            return f"不支持的操作类型: {operation}"
            
    except Exception as e:
        return f"CSV操作失败: {str(e)}"

# ✅创建Python代码执行工具
# Python代码执行工具结构化参数说明
class PythonCodeInput(BaseModel):
    py_code: str = Field(description="一段合法的 Python 代码字符串，例如 '2 + 2' 或 'x = 3\\ny = x * 2'")


@tool(args_schema=PythonCodeInput)
def python_inter(py_code):
    """
    当用户需要编写Python程序并执行时，请调用该函数。
    该函数可以执行一段Python代码并返回最终结果，需要注意，本函数只能执行非绘图类的代码，若是绘图相关代码，则需要调用fig_inter函数运行。
    """    
    g = globals()
    try:
        # 尝试如果是表达式，则返回表达式运行结果
        return str(eval(py_code, g))
    # 若报错，则先测试是否是对相同变量重复赋值
    except Exception as e:
        global_vars_before = set(g.keys())
        try:            
            exec(py_code, g)
        except Exception as e:
            return f"代码执行时报错{e}"
        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before
        # 若存在新变量
        if new_vars:
            result = {var: g[var] for var in new_vars}
            # print("代码已顺利执行，正在进行结果梳理...")
            return str(result)
        else:
            # print("代码已顺利执行，正在进行结果梳理...")
            return "已经顺利执行代码"

# ✅ 创建绘图工具
# 绘图工具结构化参数说明
class FigCodeInput(BaseModel):
    py_code: str = Field(description="要执行的 Python 绘图代码，必须使用 matplotlib/seaborn 创建图像并赋值给变量")
    fname: str = Field(description="图像对象的变量名，例如 'fig'，用于从代码中提取并保存为图片")

@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    当用户需要使用 Python 进行可视化绘图任务时，请调用该函数。

    注意：
    1. 所有绘图代码必须创建一个图像对象，并将其赋值为指定变量名（例如 `fig`）。
    2. 必须使用 `fig = plt.figure()` 或 `fig = plt.subplots()`。
    3. 不要使用 `plt.show()`。
    4. 请确保代码最后调用 `fig.tight_layout()`。
    5. 所有绘图代码中，坐标轴标签（xlabel、ylabel）、标题（title）、图例（legend）等文本内容，必须使用英文描述。

    示例代码：
    fig = plt.figure(figsize=(10,6))
    plt.plot([1,2,3], [4,5,6])
    fig.tight_layout()
    """
    # print("正在调用fig_inter工具运行Python代码...")

    current_backend = matplotlib.get_backend()
    matplotlib.use('Agg')

    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    
    # ✅ 设置图像保存路径（动态获取当前项目目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    images_dir = os.path.join(current_dir, "images")
    os.makedirs(images_dir, exist_ok=True)  # ✅ 自动创建 images 文件夹（如不存在）
    try:
        g = globals()
        exec(py_code, g, local_vars)
        g.update(local_vars)

        fig = local_vars.get(fname, None)
        if fig:
            image_filename = f"{fname}.png"
            abs_path = os.path.join(images_dir, image_filename)  # ✅ 绝对路径
            rel_path = os.path.join("images", image_filename)    # ✅ 返回相对路径（给前端用）

            fig.savefig(abs_path, bbox_inches='tight')
            return f"✅ 图片已保存，路径为: {rel_path}"
        else:
            return "⚠️ 图像对象未找到，请确认变量名正确并为 matplotlib 图对象。"
    except Exception as e:
        return f"❌ 执行失败：{e}"
    finally:
        plt.close('all')
        matplotlib.use(current_backend)


# ✅ 创建工具列表
tools = [
    file_operations,
    get_system_info,
    execute_command,
    datetime_operations,
    csv_operations,
    python_inter, 
    fig_inter, 
]

# ✅ 创建模型
model = ChatDeepSeek(model="deepseek-chat")

# ✅ 创建图 （Agent）
graph = create_react_agent(model=model, tools=tools, prompt=prompt)
