# 项目简介
本项目提供了一种面向异构表格的多源测试数据自动化提取与无损融合的解决方案，paddle_union（自动化提取）与【1】表格转置 + 【2】元素存储综合（融合）。
本项目实际用于文献中地下水重金属离子浓度的提取与存储工作，请按需调整。





# paddle_union（自动化提取）项目部署指南

## 先决条件

在开始部署之前，请确保您已经满足以下条件：

- 建议python=3.8
- 使用前需准备好GPU,下载cuDNN与CUDA,cuDNN与CUDA版本联系参考:https:\\developer.nvidia.com\\rdp\\cudnn-archive
- Paddle模型安装:https:\\www.paddlepaddle.org.cn\\
- layoutparser安装:下载whl后 pip install "...\layoutparser-0.0.0-py3-none-any.whl"
- paddleocr安装:pip install "paddleocr>=2.0.1"

- paddle安装检查:
- import paddle
- paddle.utils.run_check()

- 开始运行前，您只需将pdf文件夹路径传入--base_path，并按需求设置其他参数即可。

## 程序输出

- **合并的 Excel 文件**：程序会将最终的合并结果输出到 `base文件夹\\Excel_union` 目录下。
- **中间步骤数据**：程序运行中的中间步骤文件会保存在 `base文件夹\\Data` 目录下。
- **裁剪出的图像**：PDF 处理过程中裁剪出的图像将保存在 `base文件夹\\Data\\PDF名称\\PDF名称_results\\figure`。
- **裁剪出的表格**：PDF 处理过程中裁剪出的表格将保存在 `base文件夹\\Data\\PDF名称\\PDF名称_results\\table`。

## 错误处理

- 如果程序在运行过程中崩溃，可能是由于多次加载库导致的冲突问题。可以尝试在代码开头加入以下代码来解决：
  ```python
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

- 文件夹路径不能出现中文，若出现报错，可尝试去除空格等

## 程序运行
- 重启解释器后需要加载模型等，程序运行时间较长，但一般不会超过5分钟（指代码运行至出现提示信息）。





# paddle_union（自动化提取）脚本运行指南

该程序可以通过命令行参数传递自定义参数来运行，也可以直接使用默认参数执行。以下是脚本中各个参数的说明：

## 命令行参数

### 1. `--base_path`
- **类型**：`str`
- **默认值**：`"..."`
- **说明**：指定 PDF 文件所在的基础路径。建议使用双反斜杠 `\\` 来确保路径正确解析。
  
  **示例**：
  ```bash
  python script.py --base_path "D:\\MyPDFs\\Documents"

### 2. `--table_ok`
- **类型**：`bool`
- **默认值**：`"True`
- **说明**：是否提取 PDF 中的图像信息。如果设置为 True，程序会提取图像信息。
  
  **示例**：
  ```bash
  python script.py --figure_ok False

### 3. `--figure_ok`
- **类型**：`bool`
- **默认值**：`"True`
- **说明**：是否提取 PDF 中的图像信息。如果设置为 True，程序会提取图像信息。
  
  **示例**：
  ```bash
  python script.py --figure_ok True

### 4. `--model_threshold`
- **类型**：`float`
- **默认值**：`0.5`
- **说明**：设置模型检测的可信度阈值。值越高，检测越严格。默认值为 0.5，表示可信度达到 50% 以上的检测会被认为有效。
  
  **示例**：
  ```bash
  python script.py --model_threshold 0.5

### 5. `--mode`
- **类型**：`str`
- **默认值**：`normal`
- **说明**：选择处理模式。一般使用 normal 模式来处理论文等标准文档。还有一个 keyword 模式用于关键词提取，但该功能并不完善。
  
  **示例**：
  ```bash
  python script.py --mode "keyword"

### 6. `--base_folder`
- **类型**：`str`
- **默认值**：`base_folder`
- **说明**：当使用 keyword 模式时，指定关键词模式下存放 PDF 文件的基础路径。
  
  **示例**：
  ```bash
  python script.py --base_folder "D:\\KeywordPDFs"

### 7. `--keywords`
- **类型**：`list of str`
- **默认值**：`["Keywords"]`
- **说明**：当使用 keyword 模式时，传递一个关键词列表，用于在 PDF 中定位和裁剪相关区域。
  
  **示例**：
  ```bash
  python script.py --keywords "Introduction", "Conclusion"





# 融合项目部署指南
本项目用于文献中地下水重金属离子浓度的提取与存储工作，请按需调整。

## 表格转置

表格的表头元素可能存在于行或列，运行表格转置将其统一转化为行表头
按代码注释修改存放excel的路径即可

## 元素存储综合

批量提取表格信息，识别元素名称与其对应的信息，转化为五元组的形式，['filename', 'element', 'sample_info', 'unit', 'data']
按照提示修改路径，关键词信息（即无单位会干扰识别的元素）
