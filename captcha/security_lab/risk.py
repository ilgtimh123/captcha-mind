"""Defender-only risk scoring for the local CAPTCHA security lab.

The scorer is intentionally downstream of the solver loop: it consumes completed
telemetry and never feeds a score back to CaptchaMind during an attempt.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .features import MotionFeatures, extract_motion_features
from .trace import stable_digest


@dataclass
class RiskAssessment:
    score: float
    band: str
    reasons: List[str]
    features: Dict[str, Any]
    signature: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BotRiskScorer:
    """Small explainable baseline for defensive experiments.

    This is not intended as a production anti-bot model.  It provides stable,
    inspectable signals so the security lab can compare attempts and verify that
    telemetry plumbing works before replacing the scorer with a trained model.
    """

    def __init__(self) -> None:
        self._seen_signatures: Dict[str, int] = {}

    @staticmethod
    def _band(score: float) -> str:
        if score >= 70.0:
            return "high"
        if score >= 35.0:
            return "medium"
        return "low"

    def assess_features(self, features: MotionFeatures) -> RiskAssessment:
        score = 0.0
        reasons: List[str] = []

        # These broad rules are deliberately conservative.  Their purpose is to
        # surface suspiciously sparse or mechanically repeatable traces for
        # operator review, not to provide an evasion oracle to the solver.
        if features.event_count < 6:
            score += 25.0
            reasons.append("very sparse pointer telemetry")
        if 0.0 < features.duration_ms < 180.0:
            score += 15.0
            reasons.append("interaction completed unusually quickly")
        if features.move_count >= 6 and features.straightness >= 0.995:
            score += 12.0
            reasons.append("movement path is nearly perfectly straight")
        if features.stroke_count >= 2 and features.near_linear_stroke_fraction >= 0.75:
            score += 18.0
            reasons.append("most pointer strokes are mechanically straight")
        if features.stroke_count >= 2 and 0.0 < features.mean_stroke_points <= 10.0:
            score += 10.0
            reasons.append("pointer strokes use unusually sparse interpolation")
        if features.stroke_count >= 2 and 0.0 <= features.mean_stroke_step_cv < 0.12:
            score += 10.0
            reasons.append("within-stroke step lengths are unusually uniform")
        if (
            features.stroke_count >= 2
            and features.near_linear_stroke_fraction >= 0.5
            and features.mean_stroke_turn_rad < 0.01
        ):
            score += 8.0
            reasons.append("within-stroke heading changes are nearly absent")
        if features.move_count >= 8 and 0.0 < features.inter_event_cv < 0.025:
            score += 12.0
            reasons.append("event cadence is unusually uniform")
        if features.peak_speed_px_s >= 12000.0:
            score += 10.0
            reasons.append("very high peak pointer speed")
        if features.down_count != features.up_count:
            score += 8.0
            reasons.append("unbalanced pointer down/up sequence")
        if features.automation_webdriver:
            score += 35.0
            reasons.append("browser reports WebDriver automation")

        # Cross-attempt repetition should be robust to scheduler/timing jitter.
        # Build the signature from coarse spatial/motor geometry, not exact
        # duration or cadence, so the same scripted path remains recognizable
        # even when Chromium timing varies slightly between runs.
        signature = stable_digest({
            "moves": features.move_count,
            "downs": features.down_count,
            "ups": features.up_count,
            "path_bucket": int(features.path_length_px // 5.0),
            "displacement_bucket": int(features.displacement_px // 5.0),
            "straightness": round(features.straightness, 2),
            "direction_changes": features.direction_changes,
            "stroke_count": features.stroke_count,
            "mean_stroke_points_bucket": int(features.mean_stroke_points // 2.0),
            "near_linear_stroke_fraction": round(features.near_linear_stroke_fraction, 1),
            "stroke_step_cv": round(features.mean_stroke_step_cv, 1),
            "centroid_x_px": int(round(features.move_centroid_x)),
            "centroid_y_px": int(round(features.move_centroid_y)),
        })
        prior = self._seen_signatures.get(signature, 0)
        self._seen_signatures[signature] = prior + 1
        if prior >= 2:
            score += min(18.0, 6.0 + prior * 2.0)
            reasons.append("same coarse interaction signature repeated across attempts")

        score = max(0.0, min(100.0, score))
        return RiskAssessment(
            score=score,
            band=self._band(score),
            reasons=reasons,
            features=features.as_dict(),
            signature=signature,
        )

    def assess(
        self,
        motion_events,
        page_events=None,
    ) -> RiskAssessment:
        return self.assess_features(extract_motion_features(motion_events, page_events))
