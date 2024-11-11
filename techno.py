# We will make a simple program that writes a techno track from scratch

import wave
import random
import math

# * DO NOT TOUCH THIS STUFF OR YOU WILL BREAK THE PROGRAM * #

SR = 44100
BIT_DEPTH = 16
CHANNELS = 1

BPM = 128
SWING = 0.16
TOTAL_BEATS = 16 * 16

# * ===================================================== * #

# ---------------------------------------------------- #

# Empty 16-beat sequence : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

SEQUENCE = {
    'kick'      : [1, 0, 0, 0],
    'bass'      : [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    'hat'       : [0.1, 0.2, 1, 0.2],
    'chord'     : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0.5, 0, 0.625]
}


# ---------------------------------------------------- #
# Helper functions

def write_wave(filename: str, samples: list):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BIT_DEPTH // 8)
        wf.setframerate(SR)
        wf.writeframes(b''.join(samples))

def normal2bitdepth(samples: list):
    # Samples are in the range [-1, 1]
    # We will convert them to the range [-2^(BIT_DEPTH - 1), 2^(BIT_DEPTH - 1) - 1]

    clamped_samples = [max(min(sample, 1.0), -1.0) for sample in samples]
    new_samples = []

    for sample in clamped_samples:
        new_samples.append(int(sample * (2 ** (BIT_DEPTH - 1) - 1)))

    return new_samples

def sec2samp(sec: float):
    return int(sec * SR)

def samp2sec(samp: int):
    return float(samp) / SR

def beat2sec(beat: float):
    # 1 beat = 60 / BPM seconds
    # If you want a 16th note, beat = 0.25

    return beat * 60 / BPM

def hpf(samples: list, cutoff: float) -> list:
    # We will apply a simple high-pass filter to the samples
    # The cutoff frequency is in Hz

    # We will use a simple first-order high-pass filter
    # y[n] = x[n] - x[n-1] + a * y[n-1]

    a = math.exp(-2 * math.pi * cutoff / SR)
    y = [0] * len(samples)

    for i in range(1, len(samples)):
        y[i] = samples[i] - samples[i - 1] + a * y[i - 1]

    return y

def lpf(samples: list, cutoff: float) -> list:
    # We will apply a simple low-pass filter to the samples
    # The cutoff frequency is in Hz

    # We will use a simple first-order low-pass filter
    # y[n] = a * y[n-1] + (1 - a) * x[n]

    a = math.exp(-2 * math.pi * cutoff / SR)
    y = [0] * len(samples)

    for i in range(1, len(samples)):
        y[i] = a * y[i - 1] + (1 - a) * samples[i]

    return y

# ---------------------------------------------------- #

# Audio functions

def envelope(t: float, c: float) -> list:
    '''
    Returns an envelope in [0, 1] that decays over time
    t       - seconds
    c       - decay constant
    '''
    # We will use a simple exponential decay envelope

    t = abs(t)
    c = abs(c)
    
    # First, we will make a basic linear decay
    duration = sec2samp(t)
    env = []

    for i in range(duration):
        env.append(1 - i / duration)

    # Now we will apply the exponential decay with the following formula:
    # (exp(cx) - 1) / (exp(c) - 1)

    if c > 1e-8:
        for i in range(len(env)):
            env[i] = (2 ** (c * env[i]) - 1) / (2 ** (c) - 1)
    
    return env


def sine(freq: float, t: float) -> float:
    
    # TODO: Implement a sine wave
    
    return 0

def saw(freq: float, t: float) -> float:
    
    # TODO: Implement a saw wave

    return 0

def square(freq: float, t: float) -> float:
    
    # TODO: Implement a square wave

    return 0


def kick_drum(p: float = 1.0) -> list:
    # We will create a simple kick drum sound
    t = 0.2
    peak = 200
    base = 50

    # TODO: Make the pitch envelope for the kick
    pitch_env = None


    # TODO: map the frequency of the kick from [0, 1] -> [base, peak]
    


    # * ================================= * #

    amp_env = envelope(t, 5)

    samples = []

    for i in range(sec2samp(t)):
        
        freq = pitch_env[i]
        time = samp2sec(i)

        # TODO: Generate a sine wave using freq and time

        sample = 0.0


        # TODO: Modulate the amplitude


        # * ============================================= * #
        
        sample = math.tanh(sample * 3)
        samples.append(sample)


    return samples


def bass_drum(p: float = 1.0) -> list:
    # This will be the bass for our techno track
    t = 0.2

    amp_env = envelope(t, 10)

    samples = []

    for i in range(sec2samp(t)):

        freq = 38.89
        time = samp2sec(i)
        

        # TODO: Generate a square wave
        
        sample = 0.0


        # TODO: Modulate the amplitude


        sample = math.tanh(sample * 2)
        samples.append(sample)

    samples = lpf(samples, 200)
    samples = lpf(samples, 200)
    return samples


def hi_hat(p: float = 0.1) -> list:

    # We will make a simple hi-hat sound
    # The decay will be controlled by the parameter p

    t = 0.1

    amp_env = envelope(t, 10 * (1 - p * 0.3))
    
    samples = []

    for i in range(sec2samp(t)):
        sample = random.uniform(-1, 1) * amp_env[i] * 0.7
        samples.append(sample)

    samples = hpf(samples, 2000)
    samples = hpf(samples, 2000)
    samples = lpf(samples, 5000)

    return samples


def chord_stab(p: float = 1.0) -> list:

    t = p * 0.5

    # You can use the following table to get the frequencies of the notes
    # https://muted.io/note-frequencies/

    # TODO: Get some frequencies for the chord stab!
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0

    amp_env = envelope(t, 10)

    samples = []

    for i in range(sec2samp(t)):

        # We will use a saw wave for the chord
        sample = saw(f1, samp2sec(i))
        sample += saw(f2, samp2sec(i))
        sample += saw(f3, samp2sec(i))

        sample = math.tanh(sample * 2)

        sample *= amp_env[i]

        samples.append(sample)

    samples = hpf(samples, 200)

    lpfreq = random.uniform(250, 1000)
    samples = lpf(samples, lpfreq)

    return samples


# ---------------------------------------------------- #

# Sequence Handling

dict_map = {
    'kick'      : kick_drum,
    'bass'      : bass_drum,
    'hat'       : hi_hat,
    'chord'     : chord_stab
}

def apply_swing(beat, swing_amount):
    """
    Apply swing to a beat position.
    """
    if beat % 2 == 1:  # Apply swing to odd-numbered beats
        return beat + swing_amount
    return beat

# Pattern creation and mixing functions
def create_pattern(instrument: str, pattern: list, beats: int) -> list:
    samples = []
    beat_length = beat2sec(0.25)
    beat_length_samples = sec2samp(beat_length)
    for i in range(beats * beat_length_samples):
        samples.append(0)
    
    for beat in range(beats):
        # Apply swing to the beat position
        swung_beat = apply_swing(beat, SWING)
        
        # Calculate the actual sample position with swing
        start = int(swung_beat * beat_length_samples)
        
        pattern_idx = beat % len(pattern)
        if pattern[pattern_idx] > 0:
            sound = dict_map[instrument](pattern[pattern_idx])
            sound_length = len(sound)
            for i in range(sound_length):
                if start + i < len(samples):
                    samples[start + i] += sound[i] * pattern[pattern_idx]
    
    return samples

def mix_tracks(tracks: list) -> list:
    max_length = max(len(track) for track in tracks)
    mixed = [sum(track[i] if i < len(track) else 0 for track in tracks) for i in range(max_length)]
    mixed = [sample / len(tracks) for sample in mixed]  # Compensate for the sum

    # Normalize the track
    max_sample = max(mixed, key=abs)
    mixed = [sample / max_sample for sample in mixed]

    return [min(max(sample, -1), 1) for sample in mixed]  # Clamp values to [-1, 1]

def generate_techno_track() -> list:
    tracks = []
    for instrument, pattern in SEQUENCE.items():
        tracks.append(create_pattern(instrument, pattern, TOTAL_BEATS))
    
    return mix_tracks(tracks)


# ---------------------------------------------------- #


print("Generating techno track...")
track = generate_techno_track()

print("Converting to appropriate bit depth...")
track_bitdepth = normal2bitdepth(track)
print(min(track_bitdepth), max(track_bitdepth))

print("Writing wave file...")
# Convert to bytes directly in the write_wave function call
write_wave("techno_track.wav", [sample.to_bytes(2, byteorder='little', signed=True) for sample in track_bitdepth])

print("Track generated successfully!")