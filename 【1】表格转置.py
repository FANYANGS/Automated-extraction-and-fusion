import pandas as pd
import re
import os
import openpyxl

# 定义文件夹路径，包含需要处理的 Excel 文件
dir_path = r'C:\\'

# 定义关键词列表，用于识别行内特定元素(无单位的元素)
keyword_list = ['ph', 'cn', 'tc', 'fc', 'temp', 'depth', 'eh']


def transpose(df):
    """
    对给定的 DataFrame 进行转置。

    参数:
    df (pd.DataFrame): 需要转置的 Pandas DataFrame。

    返回:
    pd.DataFrame: 转置后的 DataFrame。
    """
    df_transposed = df.transpose()
    return df_transposed


# 遍历文件夹中的所有文件
for filename in os.listdir(dir_path):
    # 检查文件是否为 Excel 文件，过滤非 .xlsx 文件
    if not filename.endswith('.xlsx'):
        continue

    # 生成文件的完整路径
    file_path = os.path.join(dir_path, filename)

    # 加载 Excel 文件的工作簿
    book = openpyxl.load_workbook(file_path)

    # 创建一个空字典，用于存储每个工作表的数据
    sheets_data = {}

    # 遍历工作簿中的所有工作表
    for sheet in book.sheetnames:
        # 读取当前工作表的内容到 Pandas DataFrame，header=None 表示不使用第一行为列名
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)

        # 初始化标识符，检测表头所在的行
        header_row = None

        # 遍历 DataFrame 中的每一行，识别表头行
        for i, row in df.iterrows():
            # 通过正则表达式匹配特定单位（如 /, ppm, ppb）和关键词列表来识别表头
            metal_count = sum([1 for cell in row if re.search(r'\(.*(/|ppm|ppb).*\)', str(cell)) or any(
                keyword in str(cell).lower() for keyword in keyword_list)])

            # 如果该行符合表头标准，则记录该行的索引并跳出循环
            if metal_count >= 2:
                header_row = i
                break

        # 如果未找到表头，则对该工作表进行转置处理，并以 "_t" 作为后缀保存
        if header_row is None:
            df = transpose(df)
            sheets_data[sheet + "_t"] = df  # 将转置后的数据存入字典
        else:
            # 如果找到表头，则保留原始数据
            sheets_data[sheet] = df  # 将原始数据存入字典

    # 将处理后的数据写回到原 Excel 文件中，保留工作表名称
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # 遍历存储的数据字典，并将每个工作表写入 Excel 文件
        for sheet, df in sheets_data.items():
            # 写入 Excel，不保留索引和表头
            df.to_excel(writer, sheet_name=sheet, index=False, header=False)
