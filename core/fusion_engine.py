"""Fusion verdict engine and state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


WAITING = "WAITING"
FACE_DETECTED = "FACE_DETECTED"
ANALYZING = "ANALYZING"
VERDICT = "VERDICT"
RESET = "RESET"


@dataclass
class VerdictPackage:
    state: str
    risk_score: float
    time_remaining: float
    verdict: Optional[str]
    deepfake_score: float
    blink_score: float
    challenge_prompt: Optional[str] = None
    challenge_progress: str = "0/0"
    liveness_score: float = 0.5


class FusionEngine:
    def __init__(
        self,
        analysis_duration: float = 0.0,
        face_hold_duration: float = 0.0,
        waiting_face_gap_tolerance: float = 0.4,
        analyzing_face_absence_tolerance: float = 2.0,
        verified_threshold: float = 35.0,
        rejected_threshold: float = 66.0,
        min_live_samples: int = 3,
        challenge_enabled: bool = False,
        challenge_timeout: float = 4.0,
        challenge_required_successes: int = 2,
        challenge_weight: float = 0.35,
        deepfake_weight: float = 0.6,
        blink_weight: float = 0.4,
        risk_smoothing_alpha: float = 1.0,
    ) -> None:
        self.analysis_duration = analysis_duration
        self.face_hold_duration = face_hold_duration
        self.waiting_face_gap_tolerance = waiting_face_gap_tolerance
        self.analyzing_face_absence_tolerance = analyzing_face_absence_tolerance
        self.verified_threshold = verified_threshold
        self.rejected_threshold = rejected_threshold
        self.min_live_samples = max(1, int(min_live_samples))

        self.challenge_enabled = challenge_enabled
        self.challenge_timeout = max(1.0, float(challenge_timeout))
        self.challenge_required_successes = max(1, int(challenge_required_successes))
        self.challenge_weight = max(0.0, min(1.0, float(challenge_weight)))

        total_weight = max(1e-6, float(deepfake_weight) + float(blink_weight))
        self.deepfake_weight = float(deepfake_weight) / total_weight
        self.blink_weight = float(blink_weight) / total_weight
        self.risk_smoothing_alpha = max(0.0, min(1.0, float(risk_smoothing_alpha)))

        self.challenge_prompts: tuple[str, ...] = (
            "Blink twice",
            "Turn head left",
            "Turn head right",
            "Open mouth",
        )
        self.reset()

    def reset(self) -> None:
        self.state = WAITING
        self.face_seen_since: Optional[float] = None
        self.last_face_ts: Optional[float] = None
        self.analysis_start: Optional[float] = None
        self.verdict_start: Optional[float] = None
        self.deepfake_scores: list[float] = []
        self.blink_scores: list[float] = []
        self._analysis_warmup = False
        self.last_risk = 50.0
        self.last_verdict: Optional[str] = None

        self.challenge_index = -1
        self.challenge_current: Optional[str] = None
        self.challenge_started_at: Optional[float] = None
        self.challenge_successes = 0
        self.challenge_failed = False
        self.challenge_risk = 0.5
        self._last_blink_count = 0

    def _weighted_average(self, values: list[float]) -> float:
        if not values:
            return 0.5
        weights = list(range(1, len(values) + 1))
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def _compute_risk(self) -> float:
        has_deepfake = len(self.deepfake_scores) > 0
        has_blink = len(self.blink_scores) > 0

        deepfake = self._weighted_average(self.deepfake_scores)
        blink = self._weighted_average(self.blink_scores)

        if has_deepfake and has_blink:
            base_risk = (deepfake * self.deepfake_weight + blink * self.blink_weight) * 100.0
        elif has_deepfake:
            base_risk = deepfake * 100.0
        elif has_blink:
            base_risk = blink * 100.0
        else:
            base_risk = 50.0

        if self.challenge_enabled:
            return (1.0 - self.challenge_weight) * base_risk + self.challenge_weight * (self.challenge_risk * 100.0)
        return base_risk

    def _to_verdict(self, risk_score: float) -> str:
        if risk_score <= self.verified_threshold:
            return "VERIFIED"
        if risk_score >= self.rejected_threshold:
            return "REJECTED"
        return "SUSPICIOUS"

    def _next_challenge_prompt(self) -> str:
        self.challenge_index = (self.challenge_index + 1) % len(self.challenge_prompts)
        return self.challenge_prompts[self.challenge_index]

    def _start_challenge(self, now_ts: float) -> None:
        self.challenge_current = self._next_challenge_prompt()
        self.challenge_started_at = now_ts

    def _complete_challenge(self, now_ts: float) -> None:
        self.challenge_successes += 1
        self.challenge_current = None
        self.challenge_started_at = now_ts

    def _eval_challenge(
        self,
        prompt: str,
        blink_count: Optional[int],
        head_turn_signal: Optional[float],
        mouth_open_signal: Optional[float],
    ) -> bool:
        if prompt == "Blink twice":
            if blink_count is None:
                return False
            delta = max(0, int(blink_count) - int(self._last_blink_count))
            return delta >= 2

        if prompt == "Turn head left":
            return head_turn_signal is not None and float(head_turn_signal) <= -0.16

        if prompt == "Turn head right":
            return head_turn_signal is not None and float(head_turn_signal) >= 0.16

        if prompt == "Open mouth":
            return mouth_open_signal is not None and float(mouth_open_signal) >= 0.12

        return False

    def _update_challenge(
        self,
        now_ts: float,
        face_present: bool,
        blink_count: Optional[int],
        head_turn_signal: Optional[float],
        mouth_open_signal: Optional[float],
    ) -> None:
        if not self.challenge_enabled:
            self.challenge_risk = 0.5
            return

        if not face_present:
            self.challenge_risk = 0.7
            return

        if self.challenge_failed:
            self.challenge_risk = 1.0
            return

        if self.challenge_successes >= self.challenge_required_successes:
            self.challenge_risk = 0.15
            return

        if self.challenge_current is None:
            self._start_challenge(now_ts)
            self.challenge_risk = 0.55
            return

        started = self.challenge_started_at or now_ts
        if (now_ts - started) > self.challenge_timeout:
            self.challenge_failed = True
            self.challenge_risk = 1.0
            return

        if self._eval_challenge(self.challenge_current, blink_count, head_turn_signal, mouth_open_signal):
            self._complete_challenge(now_ts)
            self.challenge_risk = 0.25
        else:
            self.challenge_risk = 0.55

    def update(
        self,
        now_ts: float,
        face_present: bool,
        deepfake_score: Optional[float] = None,
        blink_pattern_score: Optional[float] = None,
        blink_count: Optional[int] = None,
        head_turn_signal: Optional[float] = None,
        mouth_open_signal: Optional[float] = None,
    ) -> VerdictPackage:
        if self.state == WAITING:
            if face_present:
                self.face_seen_since = self.face_seen_since or now_ts
                self.last_face_ts = now_ts
                if now_ts - self.face_seen_since >= self.face_hold_duration:
                    self.state = ANALYZING
                    self.analysis_start = now_ts
                    self._analysis_warmup = True
            else:
                if self.last_face_ts is None or (now_ts - self.last_face_ts) > self.waiting_face_gap_tolerance:
                    self.face_seen_since = None

        elif self.state == FACE_DETECTED:
            self.state = ANALYZING
            self.analysis_start = now_ts
            self._analysis_warmup = True

        elif self.state == ANALYZING:
            if face_present:
                self.last_face_ts = now_ts

            if not face_present and (
                self.last_face_ts is None or (now_ts - self.last_face_ts) > self.analyzing_face_absence_tolerance
            ):
                self.state = WAITING
                self.face_seen_since = None
                self.analysis_start = None
                self.deepfake_scores.clear()
                self.blink_scores.clear()
                self.challenge_current = None
                self.challenge_started_at = None
                self.challenge_successes = 0
                self.challenge_failed = False
                self.challenge_risk = 0.5
            else:
                if self._analysis_warmup:
                    self._analysis_warmup = False
                    deepfake_avg = self._weighted_average(self.deepfake_scores)
                    blink_avg = self._weighted_average(self.blink_scores)
                    if self.challenge_enabled:
                        challenge_progress = f"{self.challenge_successes}/{self.challenge_required_successes}"
                    else:
                        challenge_progress = "off"
                    return VerdictPackage(
                        state=self.state,
                        risk_score=self.last_risk,
                        time_remaining=max(0.0, self.analysis_duration - (now_ts - (self.analysis_start or now_ts))),
                        verdict=None,
                        deepfake_score=deepfake_avg,
                        blink_score=blink_avg,
                        challenge_prompt=self.challenge_current,
                        challenge_progress=challenge_progress,
                        liveness_score=blink_avg,
                    )

                self.deepfake_scores.append(float(deepfake_score) if deepfake_score is not None else 0.5)
                self.blink_scores.append(float(blink_pattern_score) if blink_pattern_score is not None else 0.5)
                self._update_challenge(
                    now_ts=now_ts,
                    face_present=face_present,
                    blink_count=blink_count,
                    head_turn_signal=head_turn_signal,
                    mouth_open_signal=mouth_open_signal,
                )
                raw_risk = self._compute_risk()
                alpha = self.risk_smoothing_alpha
                self.last_risk = (1.0 - alpha) * self.last_risk + alpha * raw_risk
                if blink_count is not None:
                    self._last_blink_count = int(blink_count)

                elapsed = now_ts - (self.analysis_start or now_ts)
                if self.analysis_duration <= 0.0 or elapsed >= self.analysis_duration:
                    self.last_verdict = self._to_verdict(self.last_risk)
                    self.state = VERDICT
                    self.verdict_start = now_ts

        elif self.state == VERDICT:
            if now_ts - (self.verdict_start or now_ts) >= 5.0:
                self.state = RESET

        elif self.state == RESET:
            self.reset()

        time_remaining = 0.0
        if self.state == ANALYZING and self.analysis_start is not None:
            time_remaining = max(0.0, self.analysis_duration - (now_ts - self.analysis_start))

        deepfake_avg = self._weighted_average(self.deepfake_scores)
        blink_avg = self._weighted_average(self.blink_scores)
        if self.challenge_enabled:
            challenge_progress = f"{self.challenge_successes}/{self.challenge_required_successes}"
        else:
            challenge_progress = "off"

        live_samples = min(len(self.deepfake_scores), len(self.blink_scores))
        live_verdict = self._to_verdict(self.last_risk) if live_samples >= self.min_live_samples else None
        final_verdict = self.last_verdict if self.state in (VERDICT, RESET) else live_verdict

        return VerdictPackage(
            state=self.state,
            risk_score=self.last_risk,
            time_remaining=time_remaining,
            verdict=final_verdict,
            deepfake_score=deepfake_avg,
            blink_score=blink_avg,
            challenge_prompt=self.challenge_current,
            challenge_progress=challenge_progress,
            liveness_score=blink_avg,
        )
