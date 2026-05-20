# encoding: utf-8

from captcha.env.base import Env
from captcha.env.image_recognition.data import load_data, get_image_path, get_tmp_file_path, get_tick_image_path, set_current_sample_path
from captcha.utils import *
from captcha.utils.image_utils import *
from captcha.data_types import EnvResetResponse, EnvResponse, Action
from typing import List, Dict, Any, Optional, Union, Callable
from PIL import Image, ImageDraw

class ImageRecognitionEnv(Env):

    def __init__(
        self,
        task_index: int
    ) -> None:
        super().__init__(
            data_load_func=load_data,
            task_index=task_index
        )
        
        # 设置当前样本路径
        current_sample_data = self.task_data
        if '_sample_path' in current_sample_data:
            set_current_sample_path(current_sample_data['_sample_path'])
        
        image_path = get_image_path(self.task_data["image"])
        tick_image_path = get_tick_image_path()  # 使用固定的tick_image路径

        self.current_state = {
            "image": image_path,
            "tick_image": tick_image_path,
            "gt_box_index": self.task_data["gt_box_index"],
            "sub_image_box_list": self.task_data['sub_image_box_list'],
            "submit_box": self.task_data["submit_box"],
            "target_object_type": self.task_data["target_object_type"]  # 新增
        }

        sub_image_states = [0] * len(self.task_data['sub_image_box_list']) # 0 for not chosen, 1 for chosen
        self.current_state["sub_image_states"] = sub_image_states
        
    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = task_index
        self.task_data = self.data[task_index]
        self.actions = []
        
        # 设置当前样本路径
        if '_sample_path' in self.task_data:
            set_current_sample_path(self.task_data['_sample_path'])
        
        image_path = get_image_path(self.task_data["image"])
        tick_image_path = get_tick_image_path()  # 使用固定的tick_image路径

        self.current_state = {
            "image": image_path,
            "tick_image": tick_image_path,
            "gt_box_index": self.task_data["gt_box_index"],
            "sub_image_box_list": self.task_data['sub_image_box_list'],
            "submit_box": self.task_data["submit_box"],
            "target_object_type": self.task_data["target_object_type"]  # 新增
        }
        sub_image_states = [0] * len(self.task_data['sub_image_box_list']) # 0 for not chosen, 1 for chosen
        self.current_state["sub_image_states"] = sub_image_states

        # 打印初始传入的图片路径和目标对象类型
        print(f"初始传入图片路径: {image_path}")
        print(f"目标对象类型: {self.task_data['target_object_type']}")
        print(f"正确答案索引: {self.task_data['gt_box_index']}")

        return EnvResetResponse(observation=[image_path])

    def get_click_box_index(self, pos: List[int]) -> int:
        sub_image_box_list = self.current_state['sub_image_box_list']
        for i, box in enumerate(sub_image_box_list):
            if is_in_box(box, pos):
                return i

        return -1

    def is_click_submit(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["submit_box"], pos)

    def set_sub_image_state(self, index: int) -> None:
        sub_image_states = self.current_state["sub_image_states"]
        sub_image_states[index] = 1 - sub_image_states[index]
        print(f"切换区域 {index} 的选择状态: {sub_image_states[index]}")

    def tick_image(self, step: int) -> str:
        sub_image_states = self.current_state["sub_image_states"]
        sub_image_box_list = self.current_state["sub_image_box_list"]
        tick_box_list = []
        for i, v in enumerate(sub_image_states):
            if v == 1:
                tick_box_list.append(sub_image_box_list[i])

        output_image_path = get_tmp_file_path("tmp_image_{}.png".format(step))
        add_to_image(
            self.current_state["image"],
            tick_box_list,
            self.current_state["tick_image"],
            output_image_path
        )

        return output_image_path

    def _draw_red_boxes_on_image(self, image_path: str, boxes: List[List[int]]) -> str:
        """
        在图像上绘制红色边界框，并保存为临时文件。
        
        Args:
            image_path: 原始图像路径
            boxes: 边界框列表 [[x1,y1,x2,y2], ...]
            
        Returns:
            带红框图像的临时文件路径
        """
        # 加载原始图像，不做模式转换
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 红色边框，线宽5
        box_color = (255, 0, 0)
        line_width = 5
        
        # 绘制所有边界框
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)
        
        # 保存到临时文件
        tmp_path = get_tmp_file_path("bounding_result.png")
        img.save(tmp_path, quality=95)
        
        print(f"生成带红框图像: {tmp_path}, boxes_count={len(boxes)}")
        
        return tmp_path

    def calc_reward(self) -> float:
        gt_box_index = self.current_state["gt_box_index"]
        sub_image_states = self.current_state["sub_image_states"]
        
        # 打印选择结果
        selected_indices = [i for i, v in enumerate(sub_image_states) if v == 1]
        print(f"模型选择的索引: {selected_indices}")
        print(f"真实的索引: {gt_box_index}")
        
        for i, v in enumerate(sub_image_states):
            if v == 1 and i not in gt_box_index:
                return 0.0

            if v == 0 and i in gt_box_index:
                return 0.0

        return 1.0

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        reward = 0.0
        done = 0
        observation = []
        
        # 处理 bounding 动作
        if action.name == 'bounding':
            boxes = action.kwargs.get("boxes", [])
            
            if not boxes:
                print("警告: bounding 动作缺少 boxes 参数")
                done = 1
                return EnvResponse(observation=observation, reward=reward, done=done)
            
            # 获取原始图像路径
            image_path = self.current_state["image"]
            
            # 绘制红框并获取临时文件路径
            bounding_image_path = self._draw_red_boxes_on_image(image_path, boxes)
            
            # 返回带红框的图像
            observation = [bounding_image_path]
            
            return EnvResponse(observation=observation, reward=reward, done=done)
        
        # 处理 click 动作
        elif action.name == 'click':
            position = action.kwargs["position"]
            index = self.get_click_box_index(position)
            if index >= 0:
                self.set_sub_image_state(index)
            elif self.is_click_submit(position):
                print("submit result!!!")
                done = 1
                reward = self.calc_reward()

            image_path = self.tick_image(len(self.actions))
            print(f"传入图片路径: {image_path}")
            return EnvResponse(observation=[image_path], reward=reward, done=done)
        
        # 不支持的动作类型
        else:
            print(f"不支持的动作类型: {action.name}")
            image_path = self.tick_image(len(self.actions))
            observation = [image_path]
            return EnvResponse(observation=observation, reward=reward, done=done)