# Project Overview
This project provides a solution for automated extraction and lossless fusion of multi-source test data for heterogeneous tables. This method is highly automated, and the fused data retains all information, supporting various modes of data retrieval.
The project is actually used for the extraction and storage of groundwater heavy metal ion concentrations in the literature, please adjust as needed.

# Automated Table Information Extraction Module Deployment Guide

## Prerequisites

Before starting the deployment, please ensure that you meet the following conditions:

- Recommended python=3.8
- Prepare GPU, download cuDNN and CUDA before use, refer to cuDNN and CUDA version: https://developer.nvidia.com/rdp/cudnn-archive
- Paddle model installation: https://www.paddlepaddle.org.cn/
- layoutparser installation: After downloading the whl, pip install "E:\Edge\layoutparser-0.0.0-py3-none-any.whl"
- paddleocr installation: pip install "paddleocr>=2.0.1"

- Paddle installation check:
  ```python
  import paddle
  paddle.utils.run_check()
  ```

- Before starting the run, you only need to pass the path of the pdf folder to --base_path and set other parameters as needed.
- !!! Folder path must not contain Chinese characters. If errors occur, try removing spaces, '-', and make filenames less than 10 letters long,etc.

## Program Output

- **Merged Excel File**: The final merged result will be output to the `base\\Excel_union` directory.
- **Intermediate Step Data**: Intermediate step files during program execution will be saved in the `base\\Data` directory.
- **Cropped Images**: Images cropped during PDF processing will be saved in `base\\Data\\PDFName\\PDFName_results\\figure`.
- **Cropped Tables**: Tables cropped during PDF processing will be saved in `base\\Data\\PDFName\\PDFName_results\\table`.

## Error Handling

- If the program crashes during execution, it may be due to conflicts caused by multiple library loads. You can try adding the following code at the beginning of the code to solve it:
  ```python
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  ```

- Folder paths cannot contain Chinese characters. If an error occurs, try removing spaces, etc.

## Program Execution
- After restarting the interpreter, models and other components need to be loaded. The program runtime is relatively long, but generally does not exceed 5 minutes (referring to the time from code execution to the appearance of prompt information).

# Automated Table Information Extraction Script Execution Guide

The program can be run by passing custom parameters through command line arguments or by directly using the default parameters. The following are the descriptions of each parameter in the script:

## Command Line Parameters

### 1. `--base_path`
- **Type**: `str`
- **Default Value**: `"F:\\OCR\\TEST"`
- **Description**: Specifies the base path where the PDF files are located. It is recommended to use double backslashes `\\` to ensure the path is correctly parsed.
  
  **Example**:
  ```bash
  python script.py --base_path "D:\\MyPDFs\\Documents"
  ```

### 2. `--table_ok`
- **Type**: `bool`
- **Default Value**: `"True"`
- **Description**: Whether to extract table information from the PDF. If set to True, the program will extract table information.
  
  **Example**:
  ```bash
  python script.py --figure_ok False
  ```

### 3. `--figure_ok`
- **Type**: `bool`
- **Default Value**: `"True"`
- **Description**: Whether to extract image information from the PDF. If set to True, the program will extract image information.
  
  **Example**:
  ```bash
  python script.py --figure_ok True
  ```

### 4. `--model_threshold`
- **Type**: `float`
- **Default Value**: `0.5`
- **Description**: Sets the confidence threshold for model detection. The higher the value, the stricter the detection. The default value is 0.5, which means that detections with a confidence level of 50% or higher will be considered valid.
  
  **Example**:
  ```bash
  python script.py --model_threshold 0.5
  ```

### 5. `--mode`
- **Type**: `str`
- **Default Value**: `normal`
- **Description**: Selects the processing mode. Generally, the normal mode is used to process standard documents such as papers. There is also a keyword mode for keyword extraction, but this function is not perfect.
  
  **Example**:
  ```bash
  python script.py --mode "keyword"
  ```

### 6. `--base_folder`
- **Type**: `str`
- **Default Value**: `base_folder`
- **Description**: When using keyword mode, specifies the base path where the PDF files are stored in keyword mode.
  
  **Example**:
  ```bash
  python script.py --base_folder "D:\\KeywordPDFs"
  ```

### 7. `--keywords`
- **Type**: `list of str`
- **Default Value**: `["Keywords"]`
- **Description**: When using keyword mode, pass a list of keywords to locate and crop relevant areas in the PDF.
  
  **Example**:
  ```bash
  python script.py --keywords "Introduction","Conclusion"
  ```

# Lossless Fusion Module Deployment Guide Based on Tuplets
This project is used for the extraction and storage of groundwater heavy metal ion concentrations in the literature, please adjust as needed.

## Table Preprocessing

The header elements of the table may exist in rows or columns. Run the table transposition to uniformly convert them to row headers. Modify the path where the excel is stored according to the code comments.

## Data Fusion Based on Sextuplets

Batch extract table information, identify element names and their corresponding information, and convert them into the form of quintuplets, ['filename', 'element', 'sample_info', 'unit', 'data']. Modify the path and keyword information (i.e., elements without units that interfere with recognition) according to the prompts.
