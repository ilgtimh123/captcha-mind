import json
import os
import glob
from typing import Any, List

# 修改这里：指定你的数据集根目录
DATASET_ROOT = "/mnt/workspace/workgroup/wangpengcheng/data/slide_puzzle/captcha_dataset"  # 改成你的数据集路径

# 缓存文件夹路径和当前样本路径
_sample_folders = None
_current_sample_path = None


def load_data() -> List[Any]:
    """加载所有样本数据"""
    global _sample_folders
    
    # 扫描所有sample文件夹
    sample_pattern = os.path.join(DATASET_ROOT, "sample*")
    _sample_folders = sorted(glob.glob(sample_pattern))
    
    if not _sample_folders:
        raise ValueError(f"在 {DATASET_ROOT} 中没有找到sample*文件夹")
    
    print(f"找到 {len(_sample_folders)} 个样本文件夹")
    
    # 加载所有样本的配置文件
    result = []
    for i, sample_folder in enumerate(_sample_folders):
        json_path = os.path.join(sample_folder, "info.json")
        
        if not os.path.exists(json_path):
            print(f"警告: 样本 {i} 缺少配置文件: {json_path}")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            # 将JSON字段映射到代码期望的格式
            transformed_data = transform_sample_data(sample_data, sample_folder)
            result.append(transformed_data)
            
        except Exception as e:
            print(f"警告: 加载样本 {i} 失败: {e}")
            continue
    
    print(f"成功加载 {len(result)} 个样本")
    return result


def transform_sample_data(sample_data: dict, sample_folder: str) -> dict:
    """将info.json的格式转换为代码期望的格式"""
    
    # 从puzzle_position_in_composed获取初始位置信息
    puzzle_pos = sample_data["puzzle_position_in_composed"]
    target_center = sample_data["target_center_in_composed"]
    
    # 计算puzzle的box（left, top, right, bottom）
    puzzle_box = [
        puzzle_pos["left"],
        puzzle_pos["top"], 
        puzzle_pos["right"],
        puzzle_pos["bottom"]
    ]
    
    # 计算target的box（假设与puzzle相同大小）
    puzzle_width = puzzle_pos["right"] - puzzle_pos["left"]
    puzzle_height = puzzle_pos["bottom"] - puzzle_pos["top"]
    
    target_box = [
        target_center["x"] - puzzle_width // 2,
        target_center["y"] - puzzle_height // 2,
        target_center["x"] + puzzle_width // 2,
        target_center["y"] + puzzle_height // 2
    ]
    
    # 构造转换后的数据结构
    transformed_data = {
        '_sample_path': sample_folder,  # 添加样本路径信息
        'image': sample_data["composed_image"],  # 主图片
        'puzzle_image': "image.png",  # 假设拼图块图片名为puzzle.png，你可以根据实际情况调整
        
        # 格式转换：from原来的puzzle位置，to目标位置
        'gt_from_box': puzzle_box,
        'gt_to_box': target_box,
        'gt_from_pos': [puzzle_pos["left"], puzzle_pos["top"]],  # 左上角位置
        'gt_to_pos': [target_box[0], target_box[1]],  # 目标左上角位置
        
        # 初始puzzle中心和目标中心（用于判断）
        'gt_from_center': [puzzle_pos["center_x"], puzzle_pos["center_y"]],
        'gt_to_center': [target_center["x"], target_center["y"]],
        
        # 提交按钮位置（如果info.json中没有，可以设置一个默认值）
        'submit_box': sample_data.get("submit_box", [700, 500, 780, 530]),  # 默认位置，可调整
        
        # 保留原始数据以备用
        'original_data': sample_data
    }
    
    return transformed_data


def get_image_path(image_name: str) -> str:
    """获取图片的完整路径"""
    global _current_sample_path
    
    if _current_sample_path is None:
        raise ValueError("当前样本路径未设置，请先调用 reset()")
    
    return os.path.join(_current_sample_path, image_name)


def set_current_sample_path(sample_path: str):
    """设置当前样本路径（给env.py调用）"""
    global _current_sample_path
    _current_sample_path = sample_path


def get_tmp_file_path(filename: str) -> str:
    """获取临时文件路径"""
    global _current_sample_path
    
    if _current_sample_path is None:
        # 回退到原始行为
        FOLDER_PATH = os.path.dirname(__file__)
        tmp_dir = os.path.join(FOLDER_PATH, 'tmp')
    else:
        tmp_dir = os.path.join(_current_sample_path, 'tmp')
        
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
        
    file_path = os.path.join(tmp_dir, filename)
    return file_path