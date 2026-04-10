#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Any, Iterator


APP_NAME = "ztime"
APP_VERSION = "0.1.0"


@dataclass(slots=True)
class Sample:
    ts: float
    value: float
    source: str


@dataclass(slots=True)
class BucketStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def push(self, value: float) -> None:
        if self.count == 0:
            self.count = 1
            self.mean = value
            self.m2 = 0.0
            self.minimum = value
            self.maximum = value
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, float]:
        result = {
            "count": self.count,
            "mean": self.mean,
            "stdev": self.stdev,
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result


@dataclass(slots=True)
class EWMABaseline:
    alpha: float
    initialized: bool = False
    mean: float = 0.0
    variance: float = 0.0

    def push(self, value: float) -> None:
        if not self.initialized:
            self.initialized = True
            self.mean = value
            self.variance = 0.0
            return
        delta = value - self.mean
        self.mean = self.alpha * value + (1.0 - self.alpha) * self.mean
        self.variance = self.alpha * (delta * delta) + (1.0 - self.alpha) * self.variance

    @property
    def stdev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def zscore(self, value: float) -> float:
        if not self.initialized:
            return 0.0
        sd = self.stdev
        if sd == 0.0:
            return 0.0
        return (value - self.mean) / sd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "mean": self.mean,
            "stdev": self.stdev,
            "variance": self.variance,
        }


@dataclass(slots=True)
class Detection:
    ts: float
    source: str
    value: float
    bucket: str
    bucket_mean: float
    bucket_stdev: float
    bucket_samples: int
    ewma_mean: float
    ewma_stdev: float
    zscore_bucket: float
    zscore_ewma: float
    severity: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EngineConfig:
    min_bucket_samples: int = 5
    history_size: int = 168
    zscore_warn: float = 2.5
    zscore_crit: float = 4.0
    ewma_alpha: float = 0.25
    freeze_on_critical: bool = True


class TimeBucketZScore:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.bucket_stats: Dict[str, BucketStats] = defaultdict(BucketStats)
        self.ewma_by_source: Dict[str, EWMABaseline] = defaultdict(lambda: EWMABaseline(alpha=self.config.ewma_alpha))
        self.history_by_bucket: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.config.history_size))
        self.last_detection: Optional[Detection] = None

    def bucket_key(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts)
        return f"{dt.weekday()}-{dt.hour:02d}"

    def classify(self, sample: Sample) -> Optional[Detection]:
        bucket = self.bucket_key(sample.ts)
        stats = self.bucket_stats[bucket]
        ewma = self.ewma_by_source[sample.source]
        value = sample.value

        bucket_z = 0.0
        ewma_z = ewma.zscore(value)

        if stats.count >= self.config.min_bucket_samples and stats.stdev > 0.0:
            bucket_z = (value - stats.mean) / stats.stdev

        severity = "healthy"
        reason = "normal"
        score = max(abs(bucket_z), abs(ewma_z))

        if score >= self.config.zscore_crit:
            severity = "critical"
            reason = "time-bucket deviation"
        elif score >= self.config.zscore_warn:
            severity = "warning"
            reason = "time-bucket elevation"

        detection = None
        if severity != "healthy":
            detection = Detection(
                ts=sample.ts,
                source=sample.source,
                value=value,
                bucket=bucket,
                bucket_mean=stats.mean,
                bucket_stdev=stats.stdev,
                bucket_samples=stats.count,
                ewma_mean=ewma.mean,
                ewma_stdev=ewma.stdev,
                zscore_bucket=bucket_z,
                zscore_ewma=ewma_z,
                severity=severity,
                reason=reason,
            )
            self.last_detection = detection

        should_push = not (severity == "critical" and self.config.freeze_on_critical)
        if should_push:
            stats.push(value)
            self.history_by_bucket[bucket].append(value)
            ewma.push(value)

        return detection

    def bucket_snapshot(self, bucket: str) -> Dict[str, Any]:
        stats = self.bucket_stats[bucket]
        history = list(self.history_by_bucket[bucket])
        return {
            "bucket": bucket,
            "stats": stats.to_dict(),
            "history_size": len(history),
            "recent": history[-10:],
        }

    def debug_state(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "bucket_count": len(self.bucket_stats),
            "last_detection": self.last_detection.to_dict() if self.last_detection else None,
        }


class InputReader:
    @staticmethod
    def from_json_lines(stream: Iterator[str], source: str) -> Iterator[Sample]:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Warning: Invalid JSON line: {e}\n")
                continue
            ts = float(obj.get("ts", time.time()))
            try:
                value = float(obj["value"])
            except (KeyError, ValueError, TypeError) as e:
                sys.stderr.write(f"Warning: Invalid value field: {e}\n")
                continue
            src = str(obj.get("source", source))
            yield Sample(ts=ts, value=value, source=src)


class DemoGenerator:
    def __init__(self, source: str, seed: int = 1337) -> None:
        self.source = source
        self.rng = random.Random(seed)

    def generate(self, points: int, start_ts: Optional[float] = None) -> Iterator[Sample]:
        ts = start_ts if start_ts is not None else time.time() - points * 3600
        for i in range(points):
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            if 0 <= hour < 6:
                base = 15.0
            elif 6 <= hour < 9:
                base = 55.0
            elif 9 <= hour < 17:
                base = 95.0
            elif 17 <= hour < 21:
                base = 70.0
            else:
                base = 30.0
            noise = self.rng.uniform(-6.0, 6.0)
            spike = 0.0
            if i in {int(points * 0.35), int(points * 0.70), int(points * 0.85)}:
                spike = self.rng.uniform(80.0, 160.0)
            value = max(0.0, base + noise + spike)
            yield Sample(ts=ts, value=value, source=self.source)
            ts += 3600


class Renderer:
    @staticmethod
    def render_text(sample: Sample, detection: Optional[Detection]) -> str:
        dt = datetime.fromtimestamp(sample.ts).strftime("%Y-%m-%d %H:%M:%S")
        if detection is None:
            return f"{dt} source={sample.source} value={sample.value:.3f} status=healthy"
        return (
            f"{dt} source={sample.source} value={sample.value:.3f} "
            f"status={detection.severity} bucket={detection.bucket} "
            f"z_bucket={detection.zscore_bucket:.3f} z_ewma={detection.zscore_ewma:.3f} "
            f"bucket_mean={detection.bucket_mean:.3f} bucket_stdev={detection.bucket_stdev:.3f}"
        )

    @staticmethod
    def render_json(sample: Sample, detection: Optional[Detection], compact: bool) -> str:
        payload = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "sample": asdict(sample),
            "detection": detection.to_dict() if detection else None,
        }
        if compact:
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Time-aware Z-score engine with hour-of-week buckets")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--points", type=int, default=240)
    parser.add_argument("--source", default="signal")
    parser.add_argument("--min-bucket-samples", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=168)
    parser.add_argument("--zscore-warn", type=float, default=2.5)
    parser.add_argument("--zscore-crit", type=float, default=4.0)
    parser.add_argument("--ewma-alpha", type=float, default=0.25)
    parser.add_argument("--freeze-on-critical", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--bucket")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.points < 1:
        raise ValueError("points must be greater than zero")
    if args.min_bucket_samples < 1:
        raise ValueError("min-bucket-samples must be at least 1")
    if args.history_size < 1:
        raise ValueError("history-size must be at least 1")
    if args.zscore_warn <= 0 or args.zscore_crit <= 0:
        raise ValueError("zscore thresholds must be greater than zero")
    if args.zscore_crit < args.zscore_warn:
        raise ValueError("zscore-crit must be greater than or equal to zscore-warn")
    if not (0.0 < args.ewma_alpha <= 1.0):
        raise ValueError("ewma-alpha must be between 0 and 1")
    if args.bucket and not args.summary:
        raise ValueError("--bucket can only be used with --summary")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        validate_args(args)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    config = EngineConfig(
        min_bucket_samples=args.min_bucket_samples,
        history_size=args.history_size,
        zscore_warn=args.zscore_warn,
        zscore_crit=args.zscore_crit,
        ewma_alpha=args.ewma_alpha,
        freeze_on_critical=args.freeze_on_critical,
    )

    engine = TimeBucketZScore(config)

    if args.demo:
        samples = DemoGenerator(args.source).generate(args.points)
    else:
        samples = InputReader.from_json_lines(sys.stdin, args.source)

    detections: List[Detection] = []

    for sample in samples:
        detection = engine.classify(sample)
        if detection is not None:
            detections.append(detection)
        
        if not args.summary:
            if args.json or args.compact_json:
                line = Renderer.render_json(sample, detection, args.compact_json)
            else:
                line = Renderer.render_text(sample, detection)
            
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    if args.summary:
        payload: Dict[str, Any] = {
            "app": APP_NAME,
            "version": APP_VERSION,
            "detections": [d.to_dict() for d in detections],
            "engine": engine.debug_state(),
        }
        if args.bucket:
            payload["bucket"] = engine.bucket_snapshot(args.bucket)
        if args.json or args.compact_json:
            if args.compact_json:
                sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            sys.stdout.write(f"{APP_NAME} {APP_VERSION}\n")
            sys.stdout.write(f"detections={len(detections)}\n")
            if args.bucket:
                bucket = engine.bucket_snapshot(args.bucket)
                sys.stdout.write(f"bucket={bucket['bucket']}\n")
                stats = bucket['stats']
                if stats.get('minimum') is not None and stats.get('maximum') is not None:
                    sys.stdout.write(
                        f"count={stats['count']} mean={stats['mean']:.3f} "
                        f"stdev={stats['stdev']:.3f} min={stats['minimum']:.3f} max={stats['maximum']:.3f}\n"
                    )
                else:
                    sys.stdout.write(
                        f"count={stats['count']} mean={stats['mean']:.3f} "
                        f"stdev={stats['stdev']:.3f}\n"
                    )
            for item in detections:
                sys.stdout.write(
                    f"{datetime.fromtimestamp(item.ts).strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{item.severity} {item.source} value={item.value:.3f} "
                    f"z_bucket={item.zscore_bucket:.3f} z_ewma={item.zscore_ewma:.3f}\n"
                )
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    main()
