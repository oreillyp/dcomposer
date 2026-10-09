from pathlib import Path

# User assets belong to the working directory, never an installed package folder.
MANIFESTS_DIR = Path.cwd() / "manifests"
DATA_DIR = Path.cwd() / "data"
PRETRAINED_DIR = Path.cwd() / "pretrained"
ASSETS_DIR = Path.cwd() / "assets"

STEMS = ["drums", "bass", "vocals", "other", "mixture"]
SAMPLE_RATE = 44_100
DURATION = 6.0
EPS = 1e-8
MIDI_TIME_RES = 100

RAW_MIDI_NOTE_TO_FINE_MIDI_NOTE = {
    22: 42,
    26: 46,
    35: 36,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 38,
    41: 43,
    42: 42,
    43: 43,
    44: 44,
    45: 47,
    46: 46,
    47: 47,
    48: 50,
    49: 49,
    50: 50,
    51: 51,
    52: 49,
    53: 51,
    54: 54,
    55: 55,
    56: 56,
    57: 49,
    58: 43,
    59: 51,
    60: 60,
    61: 60,
    62: 62,
    63: 62,
    64: 62,
    65: 65,
    66: 65,
    67: 67,
    68: 67,
    69: 67,
    70: 70,
    71: 71,
    72: 71,
    73: 67,
    74: 67,
    75: 75,
    76: 76,
    77: 76,
    78: 67,
    79: 67,
    80: 67,
    81: 67,
}

FINE_MIDI_NOTE_TO_COARSE_MIDI_NOTE = {
    36: 36,
    38: 38,
    37: 37,
    39: 39,
    42: 42,
    44: 42,
    46: 42,
    49: 49,
    51: 49,
    55: 49,
    43: 43,
    47: 43,
    50: 43,
    54: 54,
    56: 56,
    70: 70,
    60: 67,
    62: 67,
    65: 67,
    75: 67,
    76: 67,
    67: 67,
    71: 71,
}

FINE_MIDI_NOTE_TO_FINE_LABEL = {
    36: "kick",
    38: "snare",
    37: "rim",
    39: "clap",
    42: "hat_closed",
    44: "hat_pedal",
    46: "hat_open",
    49: "crash",
    51: "ride",
    55: "splash",
    43: "tom_low",
    47: "tom_mid",
    50: "tom_high",
    54: "tambourine",
    56: "cowbell",
    70: "shaker",
    60: "bongo",
    62: "conga",
    65: "timbale",
    75: "clave",
    76: "woodblock",
    67: "perc",
    71: "other",
}

COARSE_MIDI_NOTE_TO_COARSE_LABEL = {
    36: "kick",
    38: "snare",
    37: "rim",
    39: "clap",
    42: "hat",
    43: "tom",
    49: "cymbal",
    54: "tambourine",
    56: "cowbell",
    70: "shaker",
    67: "perc",
    71: "other",
}

DEFAULT_COARSE_LOUDNESS_RANGES = {
    36: (-20.0, -14.0),  # kick
    37: (-28.0, -22.0),  # rim
    38: (-28.0, -22.0),  # snare
    39: (-28.0, -22.0),  # clap
    42: (-32.0, -24.0),  # hat
    43: (-28.0, -20.0),  # tom
    49: (-36.0, -26.0),  # cymbal
    54: (-32.0, -24.0),  # tambourine
    56: (-30.0, -22.0),  # cowbell
    67: (-30.0, -22.0),  # perc
    70: (-34.0, -26.0),  # shaker
    71: (-30.0, -22.0),  # other
}

PAD_TOK = 0
BOS_TOK = 1
EOS_TOK = 2

MAX_DURATION = 10.0
MAX_ONSET_TOK = int(MAX_DURATION * MIDI_TIME_RES)

MIDI_TOKS = sorted(FINE_MIDI_NOTE_TO_FINE_LABEL.keys())
MIDI_TO_TOK = {note: i + EOS_TOK + 1 for i, note in enumerate(MIDI_TOKS)}
ONSET_TOK_OFFSET = len(MIDI_TOKS) + EOS_TOK + 1
ACOUSTIC_TOK_OFFSET = ONSET_TOK_OFFSET + MAX_ONSET_TOK + 1
