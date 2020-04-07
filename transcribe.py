import essentia
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt


def normalize(array):
    return array / np.max(array)


def extract_rhythm(audio, sr, bpm, quantization_unit=0.5, verbose=False):
    frameSize = 1024
    hopSize = 512

    # Detect onsets.
    od = es.OnsetDetection(method='hfc')
    w = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    pool = essentia.Pool()
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.hfc', od(mag, phase))

    onsets = es.Onsets()(np.array([pool['features.hfc']]), [1])

    # Determine location of beats.
    bt = es.BeatTrackerMultiFeature()
    ticks, confidence = bt(audio)

    # Now put times in terms of beats.
    events = onsets * (bpm / 60)
    beats = ticks * (bpm / 60)
    # Quantize event onsets using beats.
    event_index = 0
    beat_index = 0
    out = []
    while event_index < len(events):
        prev_beat = beats[beat_index]
        next_beat = beats[beat_index+1]
        event = events[event_index]
        if event < prev_beat:
            event_index += 1
        elif event > next_beat:
            beat_index += 1
        else:
            offset = (event - prev_beat) / (next_beat - prev_beat)
            quant_offset = round(offset / quantization_unit) * quantization_unit
            out.append(beat_index + quant_offset)
            event_index += 1

    out = np.array(out)

    if verbose:
        time = np.arange(len(audio)) / sr
        fig, (wf, ax0, ax1, ax2) = plt.subplots(4, sharex=True)
        wf.set_title('Audio')
        wf.plot(time, audio)
        ax0.set_title('Onsets')
        ax0.plot(time, np.repeat(normalize(pool['features.hfc']), hopSize)[:len(time)])
        ax0.vlines(onsets, 0, 2)
        ax1.set_title('Quantized Events')
        ax1.vlines((out + beats[0]) / (bpm / 60), 0, 1)
        ax2.set_title(f'Beats (confidence = {confidence})')
        ax2.vlines(ticks, 0, 1)
        plt.show()

    # For the moment, we're calling the first event beat 0.
    return out - out[0]

def test_extract_rhythm():
    audio = es.MonoLoader(filename='test/clap_rhythm.wav', sampleRate=44100)()
    rhythm = extract_rhythm(audio, 44100, 120, verbose=True)
    assert((rhythm == [0, 4, 5.5, 6, 8, 9.5, 10, 11, 12]).all())

if __name__ == '__main__':
    test_extract_rhythm()