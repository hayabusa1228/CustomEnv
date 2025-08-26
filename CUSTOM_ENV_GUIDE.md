# DreamerV3 カスタム環境での学習ガイド

## 概要
このドキュメントでは、DreamerV3-torchにカスタム環境を統合して学習を実行する方法を説明します。

## カスタム環境の実装

### 1. 環境ラッパーの作成

`envs/`ディレクトリに新しい環境ラッパーファイルを作成します：

```python
# envs/custom_env.py
import gymnasium as gym
import numpy as np
from . import wrappers

class CustomEnv:
    def __init__(self, name, **kwargs):
        # カスタム環境の初期化
        self._env = YourCustomEnvironment(name, **kwargs)
        self._env = wrappers.NormalizeActions(self._env)
        # 必要に応じて他のラッパーを追加
        
    @property
    def obs_space(self):
        # 観測空間の定義
        spaces = {
            'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            # 他の観測を追加可能
        }
        return spaces
    
    @property
    def act_space(self):
        # 行動空間の定義
        return self._env.action_space
    
    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action)
        return self._format_obs(obs), reward, done, truncated, info
    
    def reset(self):
        obs, info = self._env.reset()
        return self._format_obs(obs), info
    
    def _format_obs(self, obs):
        # 観測を辞書形式に変換
        return {'image': obs, 'is_first': False}
    
    def render(self):
        # 環境の可視化（TensorBoardでの動画記録用）
        # rgb_array形式でnumpy配列の画像を返す
        # (height, width, 3)
        return self._env.render(mode='rgb_array')
```

### 2. 環境登録

`envs/__init__.py`にカスタム環境を登録：

```python
# envs/__init__.py に追加
from .custom_env import CustomEnv

REGISTRY = {
    # 既存の環境...
    'custom': CustomEnv,
}
```

### 3. 設定ファイルの作成

`configs.yaml`にカスタム環境用の設定を追加：

```yaml
# configs.yaml に追加
custom:
  # 環境設定
  task: custom_task_name
  envs: {amount: 4, parallel: process}
  
  # 観測空間の設定
  encoder: {mlp_keys: '$^', cnn_keys: 'image'}
  decoder: {mlp_keys: '$^', cnn_keys: 'image'}
  
  # ハイパーパラメータ
  train_ratio: 64
  batch_size: 16
  batch_length: 64
  
  # モデル設定
  rssm: {deter: 4096, hidden: 1024, stoch: 32, classes: 32}
  
  # 学習設定
  model_lr: 1e-4
  actor_lr: 3e-5
  critic_lr: 3e-5
  
  # その他の設定
  actor_dist: normal
  imag_horizon: 15
  slow_value_target: True
  slow_target_update: 1
```

## 学習の実行

### 基本的な学習コマンド

```bash
# カスタム環境で学習を開始
python3 dreamer.py --configs custom --task custom_task_name --logdir ./logdir/custom
```

### デバッグモード

```bash
# 少ないステップ数でテスト実行
python3 dreamer.py --configs debug custom --task custom_task_name --logdir ./logdir/custom_debug
```

### パラメータの上書き

```bash
# コマンドラインから設定を上書き
python3 dreamer.py --configs custom \
  --task custom_task_name \
  --batch_size 32 \
  --model_lr 5e-4 \
  --logdir ./logdir/custom_tuned
```

## 環境の要件

カスタム環境は以下の要件を満たす必要があります：

### 必須メソッド
- `reset()`: 環境をリセットし、初期観測を返す
- `step(action)`: アクションを実行し、観測・報酬・終了フラグを返す
- `obs_space`: 観測空間のプロパティ
- `act_space`: 行動空間のプロパティ
- `render()`: 環境の可視化用画像を返す（オプション、ただし推奨）

### 観測フォーマット
観測は辞書形式で返す必要があります：
```python
{
    'image': np.array(...),      # 画像観測（オプション）
    'vector': np.array(...),      # ベクトル観測（オプション）
    'is_first': bool,            # エピソード開始フラグ
    'is_last': bool,             # エピソード終了フラグ
    'is_terminal': bool,         # 終端状態フラグ
}
```

## ラッパーの活用

`envs/wrappers.py`の既存ラッパーを活用できます：

```python
from . import wrappers

# アクション繰り返し
env = wrappers.ActionRepeat(env, repeat=2)

# 観測の正規化
env = wrappers.NormalizeObservation(env)

# アクションの正規化
env = wrappers.NormalizeActions(env)

# 報酬のクリッピング
env = wrappers.ClipReward(env, min_reward=-1, max_reward=1)

# タイムリミット
env = wrappers.TimeLimit(env, duration=1000)
```

## トラブルシューティング

### 観測空間のエラー
- `obs_space`プロパティが正しく定義されているか確認
- 返される観測の形状が定義と一致しているか確認

### 学習が進まない
- 報酬スケールの調整（`reward_scale`パラメータ）
- 学習率の調整（`model_lr`, `actor_lr`, `critic_lr`）
- バッチサイズとシーケンス長の調整

### メモリ不足
- `batch_size`を小さくする
- `batch_length`を短くする
- `parallel`環境数を減らす

## 参考実装

既存の環境実装を参考にできます：
- `envs/dmc.py`: DeepMind Control Suite
- `envs/atari.py`: Atari環境
- `envs/crafter.py`: Crafter環境
- `envs/minecraft.py`: Minecraft環境

## モニタリング

学習の進行状況を確認：

```bash
# TensorBoardで学習曲線を表示
tensorboard --logdir ./logdir

# ログファイルの確認
tail -f ./logdir/custom/train.log
```

## 高度な設定

### マルチGPU学習
```bash
python3 dreamer.py --configs custom --task custom_task_name --parallel 8 --logdir ./logdir/custom_multigpu
```

### チェックポイントからの再開
```bash
python3 dreamer.py --configs custom --task custom_task_name --from_checkpoint ./logdir/custom/latest.pt
```