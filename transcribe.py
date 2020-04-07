import essentia
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt

def normalize(array):
    return array / np.max(array)

def extract_rhythm(audio, sr, bpm, quantization_unit=0.5):
    frameSize = 1024
    hopSize = 512

    od1 = es.OnsetDetection(method='hfc')
    od2 = es.OnsetDetection(method='complex')
    w = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    pool = essentia.Pool()
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.hfc', od1(mag, phase))
        pool.add('features.complex', od2(mag, phase))

    onsets = es.Onsets()
    onsets_hfc = onsets(np.array([pool['features.hfc']]), [1])
    onsets_complex = onsets(np.array([pool['features.complex']]), [1])
    print(onsets_hfc, onsets_complex)

    # Now try the novelty curve
    weight = 'hybrid'
    s = es.Spectrum()
    freq_bands = es.FrequencyBands()

    bands_energies = []
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        bands_energies.append(freq_bands(s(w(frame))))

    novelty = es.NoveltyCurve(frameRate=44100./hopSize, weightCurveType=weight)(np.array(bands_energies))
    onsets_novelty = onsets(np.array([novelty]), [1])
    print(onsets_novelty)

    time = np.arange(len(audio)) / sr
    fig, (wf, ax0, ax1, ax2) = plt.subplots(4, sharex=True)
    wf.set_title('Audio')
    wf.plot(time, audio)
    ax0.set_title('HFC Onsets')
    ax0.plot(time, np.repeat(normalize(pool['features.hfc']), hopSize)[:len(time)])
    ax0.vlines(onsets_hfc, 0, 2)
    ax1.set_title('Complex Onsets')
    ax1.plot(time, np.repeat(normalize(pool['features.complex']), hopSize)[:len(time)])
    ax1.vlines(onsets_complex, 0, 2)
    ax2.set_title('Novelty Onsets')
    ax2.plot(time, np.repeat(normalize(novelty), hopSize)[:len(time)])
    ax2.vlines(onsets_novelty, 0, 2)
    plt.show()

def test():
    audio = es.MonoLoader(filename='test/clap_rhythm.wav', sampleRate=44100)()
    extract_rhythm(audio, 44100, 120)

test()