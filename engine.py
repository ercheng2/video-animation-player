"""Video Animation Player Engine - 视频播放 + 序列帧动画 + 状态机 + UDP通信"""
import cv2
import numpy as np
import os
import json
import socket
import threading
import time
import queue
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from PIL import Image, ImageTk


class PlayerState(Enum):
    IDLE = "idle"
    PLAYING_A = "playing_a"
    A_TO_B = "a_to_b"      # A播完，切换B的过渡瞬间
    PLAYING_B = "playing_b"
    STOPPED = "stopped"


@dataclass
class AnimationConfig:
    """序列帧动画配置"""
    path: str = ""
    loop: bool = False
    frame_duration: float = 0.1   # 每帧显示秒数
    total_frames: int = 0
    frame_files: list = field(default_factory=list)


class BackgroundVideoReader:
    """背景视频读取器 - 独立线程，持续循环"""
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: str = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def open(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        self._video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return True

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None

    def _read_loop(self):
        while self._running:
            try:
                cap = cv2.VideoCapture(self._video_path)
                if not cap.isOpened():
                    time.sleep(1)
                    continue
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        # 循环播放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._lock:
                        self._latest_frame = frame.copy()
                    # 非阻塞放入队列
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        # 丢弃旧帧
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                cap.release()
            except Exception as e:
                print(f"[VideoReader] 错误: {e}")
                time.sleep(1)

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（非阻塞）"""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            return None

    def is_running(self) -> bool:
        return self._running


class SequenceAnimator:
    """序列帧动画播放器"""
    def __init__(self):
        self.config: Optional[AnimationConfig] = None
        self._frames: list = []          # 预加载的PIL Image列表
        self._current_idx: int = 0
        self._frame_timer: float = 0.0
        self._playing = False
        self._loop = False
        self._finished = False
        self._lock = threading.Lock()
        self._last_frame: Optional[Image.Image] = None

    def load_config(self, cfg: AnimationConfig) -> bool:
        """加载序列帧配置并预加载图片"""
        if not cfg.path or not os.path.isdir(cfg.path):
            return False
        # 扫描目录下所有图片文件
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        files = sorted([
            os.path.join(cfg.path, f)
            for f in os.listdir(cfg.path)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not files:
            return False
        cfg.total_frames = len(files)
        cfg.frame_files = files
        self.config = cfg
        self._loop = cfg.loop

        # 预加载图片
        self._frames = []
        for f in files:
            try:
                img = Image.open(f).convert("RGBA")
                self._frames.append(img)
            except Exception as e:
                print(f"[Animator] 加载图片失败 {f}: {e}")
        if not self._frames:
            return False
        print(f"[Animator] 已加载 {len(self._frames)} 帧: {cfg.path}")
        return True

    def start(self):
        """开始播放"""
        with self._lock:
            self._playing = True
            self._finished = False
            self._current_idx = 0
            self._frame_timer = 0.0

    def stop(self):
        """停止播放，保留最后一帧"""
        with self._lock:
            self._playing = False
            self._finished = True

    def reset(self):
        with self._lock:
            self._playing = False
            self._finished = False
            self._current_idx = 0
            self._frame_timer = 0.0
            self._frames = []
            self.config = None

    def update(self, dt: float) -> bool:
        """更新动画状态，返回是否播放完成"""
        with self._lock:
            if not self._playing or not self._frames:
                return False

            self._frame_timer += dt
            frame_duration = self.config.frame_duration if self.config else 0.1

            if self._frame_timer >= frame_duration:
                self._frame_timer = 0
                self._current_idx += 1

                if self._current_idx >= len(self._frames):
                    if self._loop:
                        self._current_idx = 0
                    else:
                        self._current_idx = len(self._frames) - 1
                        self._playing = False
                        self._finished = True
                        return True  # 播放完成
            return False

    def get_current_frame(self) -> Optional[Image.Image]:
        """获取当前帧"""
        with self._lock:
            if not self._frames:
                return None
            if self._current_idx < len(self._frames):
                self._last_frame = self._frames[self._current_idx]
                return self._frames[self._current_idx]
            return self._last_frame

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def is_finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def current_idx(self) -> int:
        with self._lock:
            return self._current_idx

    @property
    def total_frames(self) -> int:
        with self._lock:
            return len(self._frames) if self._frames else 0


class StateMachine:
    """状态机：空闲 → 播放A → A完成 → 播放B → B完成 → 空闲"""
    def __init__(self):
        self.state = PlayerState.IDLE
        self._current_seq = "none"
        self._on_state_change: Optional[Callable] = None
        self._lock = threading.Lock()

    def set_callback(self, cb: Callable):
        self._on_state_change = cb

    def transition_to(self, new_state: PlayerState):
        with self._lock:
            old = self.state
            self.state = new_state
            if new_state == PlayerState.PLAYING_A:
                self._current_seq = "A"
            elif new_state == PlayerState.PLAYING_B:
                self._current_seq = "B"
            elif new_state == PlayerState.IDLE:
                self._current_seq = "none"
        if self._on_state_change:
            self._on_state_change(old, new_state)

    def get_state(self) -> PlayerState:
        with self._lock:
            return self.state

    def get_current_seq(self) -> str:
        with self._lock:
            return self._current_seq


class CommandHandler:
    """UDP指令接收器"""
    def __init__(self, port: int = 9999):
        self.port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self.on_command: Optional[Callable] = None
        self.on_log: Optional[Callable] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None

    def _listen_loop(self):
        while self._running:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.settimeout(2.0)
                self._sock.bind(("0.0.0.0", self.port))
                self._log(f"UDP 指令监听端口: {self.port}")

                while self._running:
                    try:
                        data, addr = self._sock.recvfrom(65535)
                        msg = data.decode("utf-8", errors="replace").strip()
                        self._log(f"收到指令: {msg} (来自 {addr})")
                        if self.on_command:
                            self.on_command(msg)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    except Exception as e:
                        self._log(f"UDP异常: {e}")
                self._sock.close()
            except OSError as e:
                self._log(f"UDP绑定失败({self.port}): {e}")
                time.sleep(3)
            except Exception as e:
                self._log(f"UDP异常: {e}")
                time.sleep(3)

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)


class PlaybackEngine:
    """播放引擎 - 整合所有组件"""
    def __init__(self):
        self.video_reader = BackgroundVideoReader()
        self.animator_a = SequenceAnimator()
        self.animator_b = SequenceAnimator()
        self.state_machine = StateMachine()
        self.cmd_handler = CommandHandler()

        self._active_animator: Optional[SequenceAnimator] = None
        self._waiting_for_b = False
        self._lock = threading.Lock()

        # 回调
        self.on_log: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None

        # 设置状态机回调
        self.state_machine.set_callback(self._state_changed)

        # 设置指令回调
        self.cmd_handler.on_command = self._handle_command

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def _state_changed(self, old: PlayerState, new: PlayerState):
        self._log(f"状态: {old.value} → {new.value}")
        if self.on_state_change:
            self.on_state_change(new)

    def _handle_command(self, msg: str):
        """解析UDP指令"""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            self._log(f"指令解析失败: {msg}")
            return

        a_path = data.get("a_path", "")
        a_loop = data.get("a_loop", False)
        a_duration = data.get("a_frame_duration", 0.1)
        b_path = data.get("b_path", "")
        b_loop = data.get("b_loop", False)
        b_duration = data.get("b_frame_duration", 0.1)

        if not a_path or not b_path:
            self._log("指令缺少必要参数: a_path / b_path")
            return

        self._log(f"加载序列A: {a_path} (loop={a_loop}), 序列B: {b_path} (loop={b_loop})")

        # 加载序列帧
        cfg_a = AnimationConfig(path=a_path, loop=a_loop, frame_duration=a_duration)
        cfg_b = AnimationConfig(path=b_path, loop=b_loop, frame_duration=b_duration)

        ok_a = self.animator_a.load_config(cfg_a)
        ok_b = self.animator_b.load_config(cfg_b)

        if not ok_a:
            self._log(f"序列A加载失败: {a_path}")
            return
        if not ok_b:
            self._log(f"序列B加载失败: {b_path}")
            return

        # 触发播放
        self._start_playback()

    def _start_playback(self):
        """开始播放A→B序列"""
        self._waiting_for_b = False
        self.animator_a.reset()
        self.animator_b.reset()
        self.animator_a.start()
        self._active_animator = self.animator_a
        self.state_machine.transition_to(PlayerState.PLAYING_A)
        self._log(f"▶ 开始播放序列A ({self.animator_a.total_frames}帧, loop={self.animator_a._loop})")

    def update(self, dt: float):
        """每帧更新（由主循环调用）"""
        state = self.state_machine.get_state()

        if state == PlayerState.PLAYING_A:
            finished = self.animator_a.update(dt)
            if finished:
                self._log("✓ 序列A播放完成")
                # 立即切换B
                self.animator_b.start()
                self._active_animator = self.animator_b
                self.state_machine.transition_to(PlayerState.PLAYING_B)
                self._log(f"▶ 开始播放序列B ({self.animator_b.total_frames}帧, loop={self.animator_b._loop})")

        elif state == PlayerState.PLAYING_B:
            finished = self.animator_b.update(dt)
            if finished:
                self._log("✓ 序列B播放完成")
                self._active_animator = None
                self.state_machine.transition_to(PlayerState.IDLE)
                self._log("⏹ 播放结束，进入空闲状态")

    def get_current_animation_frame(self):
        """获取当前动画帧（PIL Image或None）"""
        state = self.state_machine.get_state()
        if state == PlayerState.PLAYING_A:
            return self.animator_a.get_current_frame()
        elif state == PlayerState.PLAYING_B:
            return self.animator_b.get_current_frame()
        return None

    def stop_current(self):
        """强制停止当前播放"""
        self.animator_a.stop()
        self.animator_b.stop()
        self._active_animator = None
        self._waiting_for_b = False
        self.state_machine.transition_to(PlayerState.IDLE)
        self._log("⏹ 强制停止播放")

    def start(self, video_path: str, udp_port: int = 9999):
        """启动引擎"""
        # 打开视频
        if not self.video_reader.open(video_path):
            self._log(f"无法打开视频: {video_path}")
            return False
        self._log(f"背景视频已加载: {video_path} ({self.video_reader.width}x{self.video_reader.height}, {self.video_reader.fps:.1f}fps)")

        # 启动视频播放
        self.video_reader.start()

        # 启动UDP监听
        self.cmd_handler.port = udp_port
        self.cmd_handler.start()

        return True

    def shutdown(self):
        self.cmd_handler.stop()
        self.video_reader.stop()
        self.animator_a.reset()
        self.animator_b.reset()