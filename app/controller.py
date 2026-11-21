"""与 UI 无关的分拣状态机骨架。"""

import time
from enum import Enum
from typing import Callable, Dict, Optional

from .detectors import Detection
from .protocol import Command, Packet, WasteType, WASTE_NAMES
from .transport import Transport, TransportError


class SystemState(Enum):
    IDLE = "等待识别"
    WAIT_ACK = "等待下位机确认"
    EXECUTING = "下位机执行中"
    COOLDOWN = "等待物体离开"
    COMPRESSING = "压缩中"
    PAUSED = "已暂停"
    ERROR = "通信异常"


EventCallback = Callable[[str, dict], None]


class SortingController:
    def __init__(
        self,
        transport: Transport,
        event_callback: Optional[EventCallback] = None,
        stable_frames: int = 3,
        ack_timeout: float = 2.0,
        max_retries: int = 2,
        compress_after: int = 20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.event_callback = event_callback or (lambda _name, _data: None)
        self.stable_frames = stable_frames
        self.ack_timeout = ack_timeout
        self.max_retries = max_retries
        self.compress_after = compress_after
        self.clock = clock
        self.counts = {item: 0 for item in WasteType}
        self.bin_full = {item: False for item in WasteType}
        self.state = SystemState.IDLE
        self.pending = None  # type: Optional[WasteType]
        self._candidate = None  # type: Optional[WasteType]
        self._stable_count = 0
        self._missing_frames = 0
        self._last_packet = None  # type: Optional[Packet]
        self._deadline = 0.0
        self._retry_count = 0
        self._compress_sent = False
        self._sequence = 0
        self._active_sequence = None  # type: Optional[int]

    def _emit(self, name: str, **data) -> None:
        self.event_callback(name, data)

    def _set_state(self, state: SystemState) -> None:
        if state != self.state:
            self.state = state
            self._emit("state", value=state.value)

    def reset(self) -> None:
        self.counts = {item: 0 for item in WasteType}
        self.bin_full = {item: False for item in WasteType}
        self.pending = None
        self._candidate = None
        self._stable_count = 0
        self._missing_frames = 0
        self._last_packet = None
        self._retry_count = 0
        self._compress_sent = False
        self._active_sequence = None
        self._set_state(SystemState.IDLE)
        self._emit("counts", values=self.named_counts())
        self._emit("bins", values=self.named_bins())

    def named_counts(self) -> Dict[str, int]:
        return {WASTE_NAMES[key]: value for key, value in self.counts.items()}

    def named_bins(self) -> Dict[str, bool]:
        return {WASTE_NAMES[key]: value for key, value in self.bin_full.items()}

    def process_detection(self, detection: Optional[Detection]) -> None:
        """TODO 4：完成稳定帧判定、冷却消抖和分类指令触发。"""
        if detection is None:
            self._candidate = None
            self._stable_count = 0
            if self.state == SystemState.COOLDOWN:
                self._missing_frames += 1
                if self._missing_frames >= self.stable_frames:
                    self._missing_frames = 0
                    self._set_state(SystemState.IDLE)
            return

        if self.state == SystemState.COOLDOWN:
            # 目标还在画面中，继续保持冷却，不重复发送。
            self._missing_frames = 0
            return

        if self.state != SystemState.IDLE:
            return

        if self.bin_full[detection.waste_type]:
            self._missing_frames = 0
            self._set_state(SystemState.COOLDOWN)
            self._emit(
                "log",
                message="{}已满载，跳过本次分拣".format(
                    WASTE_NAMES[detection.waste_type]
                ),
            )
            return

        if detection.waste_type == self._candidate:
            self._stable_count += 1
        else:
            self._candidate = detection.waste_type
            self._stable_count = 1

        if self._stable_count >= self.stable_frames:
            waste_type = detection.waste_type
            self._candidate = None
            self._stable_count = 0
            self._send_classification(waste_type)

    def _send_packet(self, packet: Packet, wait_for_reply: bool = True) -> None:
        try:
            self.transport.send(packet)
        except TransportError as exc:
            self._set_state(SystemState.ERROR)
            self._emit("log", message="发送失败：{}".format(exc))
            return
        self._emit("tx", packet=packet.encode().hex(" ").upper())
        if wait_for_reply:
            self._last_packet = packet
            self._deadline = self.clock() + self.ack_timeout
            self._retry_count = 0

    def _send_classification(self, waste_type: WasteType) -> None:
        """TODO 5：处理满载、序号，并发送 WASTE_RESULT。"""
        if self.bin_full[waste_type]:
            self._set_state(SystemState.COOLDOWN)
            return

        self._sequence = (self._sequence + 1) & 0xFF
        if self._sequence == 0:
            self._sequence = 1

        sequence = self._sequence
        self._active_sequence = sequence
        self.pending = waste_type
        packet = Packet(
            Command.WASTE_RESULT,
            bytes((sequence, int(waste_type))),
        )
        self._send_packet(packet)
        if self.state != SystemState.ERROR:
            self._set_state(SystemState.WAIT_ACK)

    def poll(self) -> None:
        """TODO 6：轮询接收；超时后重发，超过次数进入 ERROR。"""
        for packet in self.transport.poll():
            self._emit("rx", packet=packet.encode().hex(" ").upper())
            self.handle_packet(packet)

        if self.state in (
            SystemState.WAIT_ACK,
            SystemState.EXECUTING,
            SystemState.COMPRESSING,
        ):
            if self.clock() < self._deadline:
                return

            if self._retry_count >= self.max_retries or self._last_packet is None:
                self._set_state(SystemState.ERROR)
                self._emit("log", message="等待下位机确认超时，已停止重试")
                return

            self._retry_count += 1
            self._deadline = self.clock() + self.ack_timeout
            self._emit(
                "log",
                message="等待确认超时，第 {} 次重发".format(self._retry_count),
            )
            self._send_packet(self._last_packet, wait_for_reply=False)

    def handle_packet(self, packet: Packet) -> None:
        """TODO 7：处理 ACK、ACTION_DONE、BIN_FULL 和 ERROR。"""
        if packet.command == Command.ACK:
            if len(packet.payload) < 2:
                return
            original_command, sequence = packet.payload[0], packet.payload[1]
            if (
                self._active_sequence is not None
                and sequence == self._active_sequence
                and original_command in (
                    int(Command.WASTE_RESULT),
                    int(Command.START_COMPRESS),
                )
            ):
                self._deadline = self.clock() + self.ack_timeout
                if self.state in (SystemState.WAIT_ACK, SystemState.COMPRESSING):
                    self._set_state(
                        SystemState.EXECUTING
                        if original_command == int(Command.WASTE_RESULT)
                        else SystemState.COMPRESSING
                    )
            return

        if packet.command == Command.ACTION_DONE:
            if not packet.payload:
                return
            sequence = packet.payload[0]
            if self._active_sequence is None or sequence != self._active_sequence:
                return

            completed = self.pending
            self.pending = None
            self._active_sequence = None
            self._last_packet = None
            self._retry_count = 0
            self._missing_frames = 0
            self._set_state(SystemState.COOLDOWN)

            if completed is not None:
                self.counts[completed] += 1
                self._emit("counts", values=self.named_counts())
                self._maybe_compress()
            return

        if packet.command == Command.BIN_FULL:
            if not packet.payload:
                return
            try:
                waste_type = WasteType(packet.payload[0])
            except ValueError:
                return
            self.bin_full[waste_type] = True
            self._emit("bins", values=self.named_bins())
            self._emit("log", message="{}已满载".format(WASTE_NAMES[waste_type]))
            if self.pending == waste_type or self.state == SystemState.IDLE:
                self._set_state(SystemState.COOLDOWN)
            return

        if packet.command == Command.ERROR:
            message = packet.payload.decode("utf-8", errors="replace")
            self._emit("log", message="下位机错误：{}".format(message))
            self._set_state(SystemState.ERROR)

    def _maybe_compress(self) -> None:
        """加分项：达到阈值后发送一次 START_COMPRESS。"""
        if self._compress_sent:
            return
        if sum(self.counts.values()) < self.compress_after:
            return

        self._compress_sent = True
        self._sequence = (self._sequence + 1) & 0xFF
        if self._sequence == 0:
            self._sequence = 1
        sequence = self._sequence
        self._active_sequence = sequence
        packet = Packet(Command.START_COMPRESS, bytes((sequence,)))
        self._send_packet(packet)
        if self.state != SystemState.ERROR:
            self._set_state(SystemState.COMPRESSING)