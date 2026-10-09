"""Overlapping model contexts with disjoint onset ownership."""

from .retrieval import parse_events


def windows(n_samples, sample_rate, overlap=1.0, duration=4.0):
    size = round(duration * sample_rate)
    hop = round((duration - overlap) * sample_rate)
    if n_samples <= 0 or not 0 <= overlap < duration or hop < 1:
        raise ValueError("Expected nonempty audio and overlap smaller than context")
    starts = list(range(0, max(1, n_samples - size + 1), hop))
    if starts[-1] + size < n_samples:
        starts.append(n_samples - size)
    boundaries = (
        [0] + [(a + size + b) // 2 for a, b in zip(starts, starts[1:])] + [n_samples]
    )
    return list(zip(starts, boundaries[:-1], boundaries[1:]))


def merge_events(predictions, plan, sample_rate):
    if len(predictions) != len(plan):
        raise ValueError("One prediction is required per audio window")
    events = []
    for predicted, (start, left, right) in zip(predictions, plan):
        for note, label, onset, codes in predicted:
            position = start + round(onset * sample_rate)
            if left <= position < right:
                events.append((note, label, position / sample_rate, codes))
    # Simultaneous hits and flams can be intentional; do not globally deduplicate.
    return sorted(events, key=lambda event: event[2])


def window_events(payload, count, n_vocab):
    predictions = [
        parse_events(window["tokens"][: window["length"]], count, n_vocab)
        for window in payload["windows"]
    ]
    return merge_events(predictions, payload["window_plan"], payload["sample_rate"])


def event_row(events):
    return {
        "notes": [event[0] for event in events],
        "onsets_sec": [event[2] for event in events],
        "codes": [event[3] for event in events],
    }
