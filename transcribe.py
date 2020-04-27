import essentia
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt
import itertools


def quantize(x, quantization_unit):
    return round(x / quantization_unit) * quantization_unit


def extract_rhythm(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    frame_size = 1024
    hop_size = 512

    # Detect onsets.
    od = es.OnsetDetection(method='hfc')
    w = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    onset_curve = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        mag, phase, = c2p(fft(w(frame)))
        onset_curve.append(od(mag, phase))
    onset_curve = np.array(onset_curve)

    onsets = es.Onsets()(np.array([onset_curve]), [1])

    # Determine location of beats.
    # bt = es.BeatTrackerMultiFeature(minTempo=bpm - 10, maxTempo=bpm + 10)
    # ticks, confidence = bt(audio)
    # For the moment, we'll assume the first beat is right at the start of the audio and we'll follow the tempo exactly.
    num_beats = int(np.ceil(len(audio) / sr * (bpm / 60))) + 1
    print(f'Generating {num_beats} beats.')
    ticks = np.arange(num_beats) / (bpm / 60)
    confidence = None

    # Now put times in terms of beats.
    events = onsets * (bpm / 60)
    beats = ticks * (bpm / 60)
    if verbose:
        print(events)
        print(beats)
    # Quantize event onsets using beats.
    event_index = 0
    beat_index = 0
    out = []
    while event_index < len(events):
        if verbose:
            print(event_index, beat_index, out)
        prev_beat = beats[beat_index]
        next_beat = beats[beat_index+1]
        event = events[event_index]
        if event < prev_beat:
            event_index += 1
        elif event > next_beat:
            beat_index += 1
        else:
            offset = (event - prev_beat) / (next_beat - prev_beat)
            quant_offset = quantize(offset, quantization_unit)
            out.append(beat_index + quant_offset)
            event_index += 1

    out = np.array(out)

    if verbose:
        time = np.arange(len(audio)) / sr
        fig, (wf, ax0, ax1, ax2) = plt.subplots(4, sharex=True)
        wf.set_title('Audio')
        wf.plot(time, audio)
        ax0.set_title('Onsets')
        ax0.plot(time, np.repeat(onset_curve / np.max(onset_curve), hop_size)[:len(time)])
        ax0.vlines(onsets, 0, 2)
        ax1.set_title('Quantized Events')
        ax1.vlines((out + beats[0]) / (bpm / 60), 0, 1)
        ax2.set_title(f'Beats (confidence = {confidence})')
        ax2.vlines(ticks, 0, 1)
        plt.show()

    return np.unique(out)


def freq_to_pitch(freq):
    freq = np.ascontiguousarray(freq).copy()
    valid = freq > 0
    freq[valid] = np.log2(freq[valid] / 440) * 12 + 69
    return freq


def extract_melody(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    frame_size = int(sr * quantization_unit / (bpm / 60)) // 16
    hop_size = frame_size

    audio = es.EqualLoudness(sampleRate=sr)(audio)
    replayGain = es.ReplayGain()(audio)
    factor = essentia.db2amp(replayGain + 6)
    audio = es.Scale(factor=factor)(audio)
    freqs, confidence = es.PitchMelodia(sampleRate=sr, hopSize=hop_size, frameSize=frame_size)(audio)

    pitches = freq_to_pitch(freqs)
    # HACK: quantize to C major, since we don't yet support sharps and flats anyway.
    # if I have time, I will switch this to consider pitches within each quantization_unit.
    # I think that will give better results than trying to get exact measures and then rounding them.
    key = sum((list(np.array([0, 2, 4, 5, 7, 9, 11]) + i*12) for i in range(8)), [])

    if verbose:
        fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)
        time = np.arange(len(audio)) / sr
        ax0.set_title('Input')
        ax0.plot(time, audio)
        time = np.arange(len(pitches)) / (sr / hop_size)
        ax1.set_title('Pitches')
        ax1.plot(time, pitches)
        ax2.set_title('Confidence')
        ax2.plot(time, confidence)
        ax3.set_title('Pitch contour segmentation')
        ax4.set_title('Quantized melody')

    min_duration = quantization_unit / (bpm / 60)
    notes = []
    num_beats = int(np.ceil(len(audio) / sr * (bpm / 60)))
    ticks = np.arange(num_beats * 2 + 1) / (bpm / 60) / 2
    group_start = None
    group_pitch = None
    print(ticks)
    for start, end in zip(ticks, ticks[1:]):
        print('???', start, end)
        window = pitches[int(start * sr / hop_size):int(end * sr / hop_size)]
        valid = window > 0
        window = window[valid]
        if np.sum(valid) < len(window) * 2/3:
            p = 0
        else:
            p = min(key, key=lambda p: np.sum(np.abs(p - window)))
        if np.isnan(p):
            ip = 0
        else:
            ip = int(round(p))
        if group_pitch != ip:
            if group_pitch is not None:
                group_duration = start - group_start
                if group_duration:
                    thing = (group_start * (bpm / 60), group_duration * (bpm / 60), group_pitch)
                    print('THE THING I AM APPENDING IS', thing)
                    notes.append(thing)
            group_pitch = ip
            group_start = start
        print(start, end, int(start * sr / hop_size), int(end * sr / hop_size), len(pitches), p, ip)
    if group_start:
        group_duration = start - group_start
        if group_duration:
            thing = (group_start * (bpm / 60), group_duration * (bpm / 60), group_pitch)
            print('THE THING I AM APPENDING IS', thing)
            notes.append(thing)
    print('quantized notes', notes)
    quantized_notes = notes

    if verbose:
        reconstructed = np.zeros(freqs.shape)
        for start, duration, pitch in notes:
            factor = sr / hop_size
            reconstructed[int(start * factor):int((start + duration) * factor)] = pitch
        ax3.plot(time, reconstructed)

    if verbose:
        reconstructed = np.zeros(freqs.shape)
        for start, duration, pitch in quantized_notes:
            factor = (60 / bpm) * (sr / hop_size)
            ax4.axvline(start * (60 / bpm))
            ax4.axvline((start + duration) * (60 / bpm))
            reconstructed[int(start * factor):int((start + duration) * factor)] = pitch
        ax4.plot(time, reconstructed)
        plt.show()

    return quantized_notes


def test_extract_rhythm():
    audio = es.MonoLoader(filename='test/clap_rhythm.wav', sampleRate=44100)()
    rhythm = extract_rhythm(audio, 44100, 120)
    print(rhythm)
    assert((rhythm == np.array([0, 4, 5.5, 6, 8, 9.5, 10, 11, 12]) + 4).all())


def test_extract_melody():
    # audio = es.MonoLoader(filename='test/midi_melody.wav', sampleRate=44100)()
    # melody = extract_melody(audio, 44100, 120, verbose=True)
    # TODO more testing
    # print(melody)
    # print(melody == [(0.0, 1.0, 64.0), (1.0, 1.0, 67.0), (2.0, 1.0, 69.0), (3.0, 0.5, 0), (3.5, 1.0, 64.0), (4.5, 1.0, 67.0), (5.5, 0.5, 70.0), (6.0, 1.0, 69.0), (7.0, 1.0, 0), (8.0, 1.0, 64.0), (9.0, 1.0, 67.0), (10.0, 1.0, 69.0), (11.0, 0.5, 0), (11.5, 1.0, 67.0), (12.5, 1.0, 64.0)])

    audio = es.MonoLoader(filename='test/whistle_melody.wav', sampleRate=44100)()
    melody = extract_melody(audio, 44100, 120, verbose=True)
    print(melody)


if __name__ == '__main__':
    test_extract_rhythm()
    test_extract_melody()