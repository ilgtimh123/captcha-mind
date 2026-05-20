# encoding: utf-8
import json
import os
from datetime import datetime
from captcha.agents.react_agent import ReactAgent
from captcha.env.click_order.env import ClickOrderEnv
from captcha.env.image_recognition.env import ImageRecognitionEnv
from captcha.env.slide_puzzle.env import SlidePuzzleEnv
from captcha.env.patch_select.env import PatchSelectEnv
from captcha.env.dice_count.env import DiceCountEnv
from captcha.env.connect_icon.env import ConnectIconEnv
from captcha.env.coordinates.env import CoordinateEnv
from captcha.env.dart_count.env import DartCountEnv
from captcha.env.rotation_match.env import RotationMatchEnv

# MODEL_NAME = "qwen-7b-sft"
MODEL_NAME = "qwen-7b-sft"
MAX_SAMPLES = 50 # 设置为 None 测试所有样本，设置为数字（如 10）则最多测试 10 个样本

def test_bus():
    model = MODEL_NAME
    task_instruction = ""  # 不需要复杂指令，直接用内置的目标检测prompt
    agent = ReactAgent(model, task_instruction)
    
    # 初始化环境
    env = ConnectIconEnv()
    total_images = len(env.all_images)  # 获取所有图片总数
    total_folders = len(env.folder_data)  # 获取文件夹总数
    
    print(f"开始测试 {model} 模型")
    print(f"共 {total_folders} 个文件夹，{total_images} 张图片")
    print("=" * 60)
    
    start_time = datetime.now()
    
    for task_index in range(total_images):
        # 获取当前图片信息
        current_image_path = env.all_images[task_index]
        current_folder_path = env.image_to_folder[current_image_path]
        folder_name = os.path.basename(current_folder_path)
        image_name = os.path.basename(current_image_path)
        
        print(f"\n[{task_index + 1}/{total_images}] {folder_name}/{image_name}")
        
        try:
            result = agent.solve(
                env,
                task_index,
                max_num_steps=1  # 只需要1步
            )
            
            # 计算进度
            progress = (task_index + 1) / total_images * 100
            elapsed_time = datetime.now() - start_time
            estimated_total = elapsed_time * total_images / (task_index + 1)
            remaining_time = estimated_total - elapsed_time
            
            if (task_index + 1) % 10 == 0:  # 每10张图片显示一次进度
                print(f"进度: {progress:.1f}% | 已用时: {str(elapsed_time).split('.')[0]} | 预计剩余: {str(remaining_time).split('.')[0]}")
            
        except Exception as e:
            print(f"处理 {folder_name}/{image_name} 时出错: {e}")
            continue
        
        print("-" * 40)
    
    total_time = datetime.now() - start_time
    print(f"\n🎉 所有图片测试完成！")
    print(f"总用时: {str(total_time).split('.')[0]}")
    print(f"平均每张图片: {total_time.total_seconds() / total_images:.2f}秒")
    
    # 显示汇总统计
    print(f"\n📊 结果汇总:")
    total_positive = 0
    for folder_path, _ in env.folder_data:
        folder_name = os.path.basename(folder_path)
        result_filename = f"{model.replace('/', '_').replace('-', '_')}.json"
        result_path = os.path.join(folder_path, result_filename)
        
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                positive_count = result_data.get("total_positive", 0)
                processed_count = result_data.get("total_processed", 0)
                total_positive += positive_count
                
                if processed_count > 0:
                    rate = positive_count / processed_count * 100
                    print(f"  {folder_name}: {positive_count}/{processed_count} ({rate:.1f}%)")
            except Exception as e:
                print(f"  {folder_name}: 读取结果失败 - {e}")
        else:
            print(f"  {folder_name}: 没有结果文件")
    
    overall_rate = total_positive / total_images * 100 if total_images > 0 else 0
    print(f"\n总体检测率: {total_positive}/{total_images} ({overall_rate:.1f}%)")


    
def test_connect_icon_batch():
    """批量测试所有样本"""
    model = MODEL_NAME
    # task_instruction = "Using the arrows to switch images, find same two icons connected as left image."
    task_instruction = "Use the arrow buttons to browse through images. Find the image that shows the same two icons connected in the same way as shown in the left reference image."
    
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = ConnectIconEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=7)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")

        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    
    token_stats = agent.get_token_stats()

    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")

    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "connect_icon",  # 每个函数改成对应的任务类型
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }

    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_coordinates_batch():
    """批量测试所有样本"""
    model = MODEL_NAME
    # task_instruction = "Using the arrows, move icon to the indicated position."
    # task_instruction = "There are eight directions: straight up, up-left, up-right, straight left, straight right, down-left, down-right, and straight down. Use the arrows to find the image where the animal or fruit is in the correct position. You should carefully distinguish between them, as adjacent directions are only 45 degrees apart and can easily be mistaken for the same direction."
    # task_instruction = "Use the arrows to find the image where the animal or fruit is in the correct position."
    task_instruction = "Use the arrow buttons to browse through images. Find the image where the animal or fruit is in the correct position."
    
    # task_instruction = "Use the left and right arrows to browse through a sequence of images. In each image, an apple is placed on one of the chairs in a grid. Your goal is to find the image where the apple is located at the exact same row and column as shown on the left. Once you find the correct image, click SUBMIT."
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = CoordinateEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=7)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")

                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "coordinates",  # 每个函数改成对应的任务类型
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }

    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_dart_count_batch():
    """批量测试所有样本"""
    model = MODEL_NAME
    task_instruction = "Use the arrows to find the image where all the darts add up to the target number shown on the left."
    # task_instruction = "Use the left and right arrows to scroll through the images. Find the one where the sum of all the darts equals the target number shown on the left, then tap SUBMIT."

    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = DartCountEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=7)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "dart_count",  # 每个函数改成对应的任务类型
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }

    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_rotation_match():
    """批量测试所有样本"""
    model = MODEL_NAME
    task_instruction = "Use the arrows to find the image where the object faces the same direction as the reference."
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = RotationMatchEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=10)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "rotation_match",  # 每个函数改成对应的任务类型
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }

    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_dice_count_batch():
    """批量测试所有样本"""
    model = MODEL_NAME
    task_instruction = "Sum up the numbers on all the dice"
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = DiceCountEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试骰子计数，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=3)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 骰子计数批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "dice_count",  # 每个函数改成对应的任务类型
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }

    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_click_order_batch():
    """批量测试所有click_order样本"""
    model = MODEL_NAME
    task_instruction = "Click the icons in the correct order as shown in the sequence at the bottom left of the image."
    # task_instruction = "Look at the icon sequence shown in the bottom bar (the reference area). Then locate and click the matching icons in the upper area, following the exact order shown in the bottom sequence. Important: Click the icons in the UPPER AREA, not the ones in the bottom bar. The bottom bar only shows you the correct order."
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = ClickOrderEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=6)  # click_order需要更多步数
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "click_order",
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }
    
    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_slide_puzzle_batch():
    """批量测试所有slide puzzle样本"""
    model = MODEL_NAME
    task_instruction = "Drag the slider component to the correct position."
    # task_instruction = "Drag the slider on the left to the correct position in the gap on the right."
    agent = ReactAgent(model, task_instruction)
    
    # 先创建环境获取总样本数
    env = SlidePuzzleEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试slide puzzle，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: task_{task_index}")
            
            result = agent.solve(env, task_index, max_num_steps=3)
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "slide_puzzle",
        "model": model,
        "task_instruction": task_instruction,
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }
    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_image_recognition_batch():
    """批量测试所有样本"""
    model = MODEL_NAME
    
    # 先创建环境获取总样本数和构建动态指令
    env = ImageRecognitionEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples} (sample1801-sample2000)")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            print(f"当前样本: sample{1801 + task_index}")  # 显示实际的样本编号
            
            # 获取当前样本的目标对象类型，构建动态指令
            current_task_data = env.data[task_index]
            target_object_type = current_task_data["target_object_type"]
            task_instruction = f"select all images containing {target_object_type}"
            
            print(f"目标对象: {target_object_type}")
            print(f"指令: {task_instruction}")
            
            # 为每个样本创建新的Agent（因为指令不同）
            agent = ReactAgent(model, task_instruction)
            
            result = agent.solve(env, task_index, max_num_steps=10)  # image_recognition可能需要更多步骤
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "image_recognition",
        "model": model,
        "sample_range": "1801-2000",
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }
    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results

def test_patch_select_batch():
    """批量测试所有样本 (sample_1801 到 sample_2000)"""
    model = MODEL_NAME
    
    # 先创建环境获取总样本数
    env = PatchSelectEnv(task_index=0)
    total_samples = len(env.data)
    total_samples = min(total_samples, MAX_SAMPLES) if MAX_SAMPLES else total_samples
    
    print(f"开始批量测试，总样本数: {total_samples}")
    
    results = []
    success_count = 0
    
    for task_index in range(total_samples):
        try:
            print(f"\n测试进度: {task_index + 1}/{total_samples}")
            
            # 获取当前样本的目标对象类型，动态生成instruction
            current_task_data = env.data[task_index]
            target_object_type = current_task_data.get("target_object_type", "objects")
            task_instruction = f"select all squares of {target_object_type}"
            
            # 从样本路径获取样本名称
            sample_path = current_task_data.get('_sample_path', f'task_{task_index}')
            sample_name = os.path.basename(sample_path) if sample_path else f'task_{task_index}'
            
            print(f"当前样本: {sample_name}")
            print(f"目标对象: {target_object_type}")
            print(f"指令: {task_instruction}")
            
            # 为每个样本创建新的Agent（因为instruction不同）
            agent = ReactAgent(model, task_instruction)
            
            result = agent.solve(env, task_index, max_num_steps=10)  # patch_select可能需要更多步骤
            results.append(result.reward)
            
            if result.reward > 0:
                success_count += 1
                print(f"✅ 成功! reward = {result.reward}")
            else:
                print(f"❌ 失败! reward = {result.reward}")
            print(f"当前成功率: {success_count}/{task_index + 1} ({success_count/(task_index+1):.2%})")
                
        except Exception as e:
            print(f"💥 错误: {str(e)}")
            results.append(0.0)
    token_stats = agent.get_token_stats()
    
    # 统计结果
    avg_reward = sum(results) / len(results) if results else 0
    success_rate = success_count / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"📊 批量测试完成!")
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {success_count}")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 保存结果到json文件
    os.makedirs("test_results", exist_ok=True)
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "patch_select",
        "model": model,
        "sample_range": "sample_1801_to_sample_2000",
        "total_samples": total_samples,
        "success_count": success_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "token_usage": token_stats,
        "detailed_results": results
    }
    
    filename = f"test_results/{result_data['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"📁 结果已保存到: {filename}")
    
    return results


def main():


    # test_bus()
    
    # 改为批量测试
    # test_patch_select_batch()
    # test_click_order_batch()
    # test_rotation_match()

    # test_connect_icon_batch()
    # test_image_recognition_batch()
    # test_dart_count_batch()
    # test_click_order_batch()
    test_coordinates_batch()
    # test_slide_puzzle_batch()
    # test_dice_count_batch()
    # test_rotation_match()

if __name__ == '__main__':
    main()
