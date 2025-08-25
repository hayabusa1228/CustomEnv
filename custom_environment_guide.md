# カスタム環境をDreamerV3に追加する方法

このドキュメントでは、DreamerV3-PyTorchでカスタムGymnasium環境を使用するための手順を説明します。

## 概要

DreamerV3でカスタム環境を使用するには、以下の手順が必要です：

1. カスタム環境の実装
2. 環境ラッパーの作成
3. 設定ファイルの更新
4. dreamer.pyの更新
5. テストとデバッグ

## 1. カスタム環境の実装

### 基本要件

カスタム環境は以下の要件を満たす必要があります：

- `gymnasium.Env`を継承
- DreamerV3用の観測辞書形式を返す
- 適切な`action_space`と`observation_space`を定義

### 観測空間の形式

DreamerV3は以下の形式の観測を期待します：

```python
observation_space = spaces.Dict({
    'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
    'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
    'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
    'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
})
```

### 環境実装例

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # アクション空間の定義
        self.action_space = spaces.Discrete(4)  # または spaces.Box(...)
        
        # 観測空間の定義
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
        })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 環境状態の初期化
        # ...
        
        obs = self._get_obs()
        obs['is_first'] = np.array([1], dtype=np.uint8)
        obs['is_last'] = np.array([0], dtype=np.uint8)  
        obs['is_terminal'] = np.array([0], dtype=np.uint8)
        
        return obs, {}
    
    def step(self, action):
        # アクションを実行
        # ...
        
        # 報酬と終了判定を計算
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        done = terminated or truncated
        
        obs = self._get_obs()
        obs['is_first'] = np.array([0], dtype=np.uint8)
        obs['is_last'] = np.array([1 if done else 0], dtype=np.uint8)
        obs['is_terminal'] = np.array([1 if terminated else 0], dtype=np.uint8)
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        # 観測画像を生成（64x64x3のRGB画像）
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        # ... 画像の内容を設定 ...
        
        return {
            'image': image,
            'is_first': np.array([0], dtype=np.uint8),
            'is_last': np.array([0], dtype=np.uint8),
            'is_terminal': np.array([0], dtype=np.uint8),
        }
```

## 2. 環境ラッパーの作成

`envs/custom_gym.py`ファイルに環境のファクトリ関数を作成します：

```python
def make_custom_env(task, **kwargs):
    """カスタム環境を作成する関数"""
    
    if task == 'your_custom_env':
        env = YourCustomEnv(**kwargs)
    elif task == 'another_custom_env':
        env = AnotherCustomEnv(**kwargs)
    else:
        raise ValueError(f"Unknown custom environment: {task}")
    
    # DreamerV3用のラッパーを適用
    env = GymWrapper(env)
    return env

def make_env(config, mode, id):
    """DreamerV3統合用の関数"""
    task = config.task
    env = make_custom_env(task)
    
    # 標準ラッパーを適用
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    env = wrappers.UUID(env)
    
    if config.reward_EMA:
        env = wrappers.RewardObs(env)
    
    return env
```

## 3. 設定ファイルの更新

`configs.yaml`にカスタム環境用の設定を追加します：

```yaml
your_custom_env:
  task: your_custom_env
  steps: 1e5                    # 学習ステップ数
  action_repeat: 1              # アクション繰り返し
  envs: 4                       # 並列環境数
  train_ratio: 512              # 学習頻度
  video_pred_log: true          # ビデオログを有効化
  time_limit: 1000              # エピソード制限
  actor: {dist: 'onehot', std: 'none'}  # 離散アクション用
  # actor: {dist: 'normal'}     # 連続アクション用
  encoder: {mlp_keys: '$^', cnn_keys: 'image'}
  decoder: {mlp_keys: '$^', cnn_keys: 'image'}
```

### 設定パラメータの説明

- `steps`: 学習の総ステップ数
- `action_repeat`: 同一アクションを繰り返す回数
- `envs`: 並列で実行する環境の数
- `train_ratio`: データ収集に対する学習の頻度
- `time_limit`: 1エピソードの最大ステップ数
- `actor.dist`: アクション分布（`'onehot'`=離散、`'normal'`=連続）

## 4. dreamer.pyの更新

`dreamer.py`の`make_env`関数にカスタム環境のケースを追加：

```python
def make_env(config, mode, id):
    # 既存のコード...
    
    elif suite == "custom":
        import envs.custom_gym as custom_gym
        env = custom_gym.make_env(config, mode, id)
    else:
        raise NotImplementedError(suite)
    
    # 既存のラッパー適用コード...
```

## 5. 依存関係の追加

必要に応じて`requirements.txt`に依存関係を追加：

```txt
gymnasium>=0.28.0
opencv-python>=4.7.0.72  # 画像処理用
```

## 6. テスト方法

### 基本テスト

まず環境が正しく動作することを確認：

```python
# test_env.py
from envs.custom_gym import make_custom_env

env = make_custom_env('your_custom_env')
obs = env.reset()
print(f"Observation keys: {obs.keys()}")
print(f"Image shape: {obs['image'].shape}")

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    if done:
        break
```

### デバッグモードでの学習テスト

```bash
python dreamer.py --configs debug your_custom_env --task your_custom_env --logdir ./logdir/debug_custom
```

### 本格的な学習の実行

```bash
python dreamer.py --configs your_custom_env --task your_custom_env --logdir ./logdir/your_custom_env
```

## 7. よくある問題とトラブルシューティング

### 観測空間の問題

**エラー**: `KeyError: 'image'`
**解決**: 観測辞書に`'image'`キーが含まれていることを確認

**エラー**: `ValueError: observation shape mismatch`
**解決**: 画像の形状が`(64, 64, 3)`になっていることを確認

### アクション空間の問題

**エラー**: `AttributeError: 'Discrete' object has no attribute 'n'`
**解決**: 離散アクション空間では`actor: {dist: 'onehot', std: 'none'}`を使用

### メモリ使用量の問題

- `envs`パラメータを小さくする（1〜4）
- `batch_size`を小さくする
- `video_pred_log: false`にしてメモリを節約

### 学習が進まない場合

- 報酬設計を見直す（スパースすぎる報酬は学習を困難にする）
- `time_limit`を適切に設定する
- エピソードの長さを調整する

## 8. 実用的なTips

### 画像観測の作成

環境状態を64x64のRGB画像に変換する方法：

```python
def state_to_image(self):
    """環境状態を画像に変換"""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # 背景色を設定
    image[:, :] = [50, 50, 50]  # グレー
    
    # オブジェクトを描画
    # 例：エージェントを青い四角で表現
    agent_x, agent_y = self.agent_position
    x_pixel = int(agent_x * 64 / self.world_size)
    y_pixel = int(agent_y * 64 / self.world_size)
    
    image[y_pixel-2:y_pixel+2, x_pixel-2:x_pixel+2] = [0, 0, 255]  # 青
    
    return image
```

### 既存Gym環境のラップ

既存のGym環境を画像観測に変換：

```python
class GymToImageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'is_first': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_last': spaces.Box(0, 1, (1,), dtype=np.uint8),
            'is_terminal': spaces.Box(0, 1, (1,), dtype=np.uint8),
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs, is_first=True), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._convert_obs(obs, is_last=done, is_terminal=terminated), reward, terminated, truncated, info
    
    def _convert_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        # 元の観測をレンダリングまたは変換して画像を作成
        image = self.env.render()  # またはカスタム変換
        if image.shape != (64, 64, 3):
            import cv2
            image = cv2.resize(image, (64, 64))
        
        return {
            'image': image.astype(np.uint8),
            'is_first': np.array([1 if is_first else 0], dtype=np.uint8),
            'is_last': np.array([1 if is_last else 0], dtype=np.uint8),
            'is_terminal': np.array([1 if is_terminal else 0], dtype=np.uint8),
        }
```

## 9. チェックリスト

カスタム環境を追加する際のチェックリスト：

- [ ] 環境クラスが`gymnasium.Env`を継承している
- [ ] `action_space`と`observation_space`が正しく定義されている
- [ ] 観測辞書に必要なキー（`image`, `is_first`, `is_last`, `is_terminal`）が含まれている
- [ ] 画像観測が`(64, 64, 3)`の形状である
- [ ] `envs/custom_gym.py`にファクトリ関数を追加
- [ ] `configs.yaml`に環境設定を追加
- [ ] `dreamer.py`に環境のケースを追加
- [ ] 基本的なテストを実行済み
- [ ] デバッグモードでの学習テストを実行済み

このガイドに従うことで、DreamerV3でカスタムGymnasium環境を使用できるようになります。