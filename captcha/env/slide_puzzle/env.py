# encoding: utf-8

from captcha.env.base import Env
from captcha.env.slide_puzzle.data import load_data, get_image_path, get_tmp_file_path, set_current_sample_path
from captcha.utils import *
from captcha.utils.image_utils import *
from captcha.data_types import EnvResetResponse, EnvResponse, Action
from captcha.movement import MovementPlanner, action_to_movement_plan

from typing import List, Dict, Any, Optional, Union, Callable, Tuple


class SlidePuzzleEnv(Env):

    def __init__(
        self,
        task_index: int
    ) -> None:
        super().__init__(
            data_load_func=load_data,
            task_index=task_index
        )
        
        self.tolerance = 15
        self.movement_planner = MovementPlanner(
            interpolation_steps=8,
            move_duration_ms=320,
        )
        self.last_movement_plan = None
        
        # 设置当前样本路径
        current_sample_data = self.task_data
        if '_sample_path' in current_sample_data:
            set_current_sample_path(current_sample_data['_sample_path'])
            
        image_path = get_image_path(self.task_data["image"])
        puzzle_image_path = get_image_path(self.task_data["puzzle_image"])

        gt_from_width = self.task_data["gt_from_box"][2] - self.task_data["gt_from_box"][0]
        gt_from_height = self.task_data["gt_from_box"][3] - self.task_data["gt_from_box"][1]

        gt_to_width = self.task_data["gt_to_box"][2] - self.task_data["gt_to_box"][0]
        gt_to_height = self.task_data["gt_to_box"][3] - self.task_data["gt_to_box"][1]

        assert gt_from_width == gt_to_width and gt_from_height == gt_to_height, "from box and to box should have equal size!"

        self.current_state = {
            "image": image_path,
            "puzzle_image": puzzle_image_path,
            "gt_from_box": self.task_data["gt_from_box"],
            "gt_to_box": self.task_data["gt_to_box"],
            "submit_box": self.task_data["submit_box"],
            "puzzle_pos": self.task_data["gt_from_pos"] # left top point position
        }

        
    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = task_index
        self.task_data = self.data[task_index]
        self.actions = []
        self.last_movement_plan = None

        # 设置当前样本路径
        if '_sample_path' in self.task_data:
            set_current_sample_path(self.task_data['_sample_path'])

        image_path = get_image_path(self.task_data["image"])
        
        self.current_state = {
            "image": image_path,
            "gt_from_box": self.task_data["gt_from_box"],
            "gt_to_box": self.task_data["gt_to_box"],
            "submit_box": self.task_data["submit_box"],
            "puzzle_pos": self.task_data["gt_from_pos"]
        }
        
        # 打印初始传入的图片路径和拖动信息
        print(f"传入图片路径: {image_path}")
        print(f"拖动起始位置: {self.task_data['gt_from_pos']} (左上角)")
        print(f"拖动目标区域: {self.task_data['gt_to_box']} (left, top, right, bottom)")
        print(f"容错范围: {self.tolerance} 像素")
        
        # 直接返回原始图片，不生成tmp图片
        return EnvResetResponse(observation=[image_path])

    def get_puzzle_size(self) -> Tuple[int]:
        gt_from_width = self.task_data["gt_from_box"][2] - self.task_data["gt_from_box"][0]
        gt_from_height = self.task_data["gt_from_box"][3] - self.task_data["gt_from_box"][1]

        return (gt_from_width, gt_from_height) 

    def get_current_puzzle_box(self) -> Tuple[int]:
        puzzle_width, puzzle_height = self.get_puzzle_size()
        x, y = self.current_state["puzzle_pos"]
        return (x, y, x + puzzle_width, y + puzzle_height)

    def is_click_submit(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["submit_box"], pos)

    def add_puzzle_to_image(self, step: int) -> str:
        output_image_path = get_tmp_file_path("tmp_image_{}.png".format(step))
        puzzle_pos = self.current_state["puzzle_pos"]
        puzzle_width, puzzle_height = self.get_puzzle_size()
        box = [puzzle_pos[0], puzzle_pos[1], puzzle_pos[0] + puzzle_width, puzzle_pos[1] + puzzle_height]
        add_to_image(
            self.current_state["image"],
            [box],
            self.current_state["puzzle_image"],
            output_image_path
        )

        return output_image_path

    def calc_reward(self) -> float:
        gt_to_box = self.current_state["gt_to_box"]
        puzzle_width, puzzle_height = self.get_puzzle_size()
        gt_lt = (gt_to_box[0], gt_to_box[1])
        x, y = self.current_state["puzzle_pos"]
        
        # 打印选择结果
        print(f"模型选择的位置: ({x}, {y}), 真实的位置: {gt_lt}")
        print(f"位置差异: x差={abs(gt_lt[0] - x)}, y差={abs(gt_lt[1] - y)}, 容差={self.tolerance}")
        
        if abs(gt_lt[0] - x) <= self.tolerance and abs(gt_lt[1] - y) <= self.tolerance:
            return 1.0
        else:
            return 0.0

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        reward = 0.0
        done = 1  # 只需要一次拖动就结束
        observation = []
        
        if action.name == 'drag':
            from_pos, to_pos = action.kwargs["from"], action.kwargs["to"]

            # Expand the high-level drag into deterministic benchmark movement
            # primitives. The environment stores the plan for inspection/replay;
            # it does not control an operating-system pointer.
            self.last_movement_plan = action_to_movement_plan(
                action,
                self.movement_planner,
            )
            print(
                f"Movement plan generated: "
                f"{len(self.last_movement_plan.steps)} steps"
            )
            
            # 直接计算奖励，对比拖动的to_pos和真实目标位置
            gt_to_box = self.current_state["gt_to_box"]
            gt_target_center = (
                (gt_to_box[0] + gt_to_box[2]) // 2,  # 目标中心x
                (gt_to_box[1] + gt_to_box[3]) // 2   # 目标中心y
            )
            
            # 计算拖动目标和真实目标的距离
            distance_x = abs(to_pos[0] - gt_target_center[0])
            distance_y = abs(to_pos[1] - gt_target_center[1])
            
            print(f"模型拖动目标: {to_pos}")
            print(f"真实目标中心: {gt_target_center}")
            print(f"距离差异: x差={distance_x}, y差={distance_y}, 容差={self.tolerance}")
            
            if distance_x <= self.tolerance and distance_y <= self.tolerance:
                reward = 1.0
                print("✅ 拖动正确!")
            else:
                reward = 0.0
                print("❌ 拖动错误!")
                
        else:
            print(f"无效动作: {action.name}")
            
        return EnvResponse(observation=observation, reward=reward, done=done)
