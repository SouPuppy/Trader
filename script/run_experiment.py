#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union
import questionary
from questionary import Style

# 项目配置
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# 极简风格配置
PLAIN_STYLE = Style([
    ('qmark', 'bold'),       
    ('question', 'bold'),    
    ('answer', 'bold'),      
    ('pointer', 'bold'),     
    ('highlighted', 'bold'), 
    ('selected', 'bold'),    
])

def find_experiments() -> List[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    experiments = []
    for item in sorted(EXPERIMENTS_DIR.iterdir()):
        if item.is_dir() and (item / "main.py").exists():
            experiments.append(item)
    return experiments

def find_series_experiments(series_prefix: str) -> List[Path]:
    """
    查找指定系列的所有实验
    
    Args:
        series_prefix: 系列前缀，例如 "6." 表示 6.x 系列
        
    Returns:
        该系列的所有实验路径列表，按名称排序
    """
    experiments = find_experiments()
    series_experiments = []
    for exp in experiments:
        if exp.name.startswith(series_prefix):
            series_experiments.append(exp)
    # 按名称排序，确保按顺序运行
    series_experiments.sort(key=lambda x: x.name)
    return series_experiments

def run_experiment(exp_dir: Union[Path, str]):
    # [修复点 1] 防御性编程：强制转换为 Path 对象
    # 无论传入的是字符串还是 Path 对象，这里统一转为 Path
    exp_path = Path(exp_dir)
    main_py = exp_path / "main.py"
    
    cmd = ["poetry", "run", "python", str(main_py)]
    print(f"> {' '.join(cmd)}\n")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=False)
    except KeyboardInterrupt:
        print("\nInterrupt")

def run_series_experiments(series_prefix: str):
    """
    运行指定系列的所有实验
    
    Args:
        series_prefix: 系列前缀，例如 "6." 表示 6.x 系列
    """
    series_experiments = find_series_experiments(series_prefix)
    
    if not series_experiments:
        print(f"No experiments found for series '{series_prefix}'")
        return
    
    print(f"\n{'='*60}")
    print(f"Running {len(series_experiments)} experiments in series '{series_prefix}'")
    print(f"{'='*60}\n")
    
    for i, exp in enumerate(series_experiments, 1):
        print(f"\n[{i}/{len(series_experiments)}] Running: {exp.name}")
        print("-" * 60)
        run_experiment(exp)
        print("-" * 60)
    
    print(f"\n{'='*60}")
    print(f"Completed running {len(series_experiments)} experiments in series '{series_prefix}'")
    print(f"{'='*60}\n")

def main():
    while True:
        experiments = find_experiments()
        
        choices = []
        if not experiments:
             print(f"No experiments found in {EXPERIMENTS_DIR}")
             # 如果没实验，只显示退出
        else:
            # 正常添加实验选项
            for e in experiments:
                choices.append(questionary.Choice(title=e.name, value=e))
        
        # 添加运行系列实验的选项
        SERIES_TOKEN_PREFIX = "___SERIES___"
        series_6_experiments = find_series_experiments("6.")
        if series_6_experiments:
            series_6_names = [e.name for e in series_6_experiments]
            series_6_title = f"Run 6.x series ({len(series_6_experiments)} experiments: {', '.join([n.split()[0] for n in series_6_names])})"
            choices.append(questionary.Choice(title=series_6_title, value=f"{SERIES_TOKEN_PREFIX}6."))
        
        # [修复点 2] 使用明确的字符串标记作为退出值，避免 None 的歧义
        EXIT_TOKEN = "___EXIT___"
        choices.append(questionary.Choice(title="Exit", value=EXIT_TOKEN))

        selected = questionary.select(
            "Select experiment:",
            choices=choices,
            style=PLAIN_STYLE,
            pointer=">",
            qmark="?",
            use_indicator=False
        ).ask()

        # [修复点 3] 严格的退出判断逻辑
        if selected is None or selected == EXIT_TOKEN:
            sys.exit(0)
        
        # 检查是否是运行系列实验的选项
        if isinstance(selected, str) and selected.startswith(SERIES_TOKEN_PREFIX):
            series_prefix = selected.replace(SERIES_TOKEN_PREFIX, "")
            run_series_experiments(series_prefix)
        else:
            # 运行单个实验
            run_experiment(selected)
        print() 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)