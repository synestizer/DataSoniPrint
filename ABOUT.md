# DataSoniPrint

**Inclusive Data Sonification & 3D Printing from LIGO Gravitational Wave Data**

---

## Aim

Scientific data is overwhelmingly presented as visual graphs and charts — formats
that exclude people with visual impairments, and offer nothing tactile or auditory
for those with hearing aids or different sensory needs.

**DataSoniPrint** converts real LIGO gravitational wave observations into three
parallel, accessible output modalities:

| Output           | Modality  | Who benefits                                           |
|------------------|-----------|--------------------------------------------------------|
| **Sonification** | Audio     | Visually impaired users experience data through sound  |
| **Spectrogram**  | Visual    | Traditional visual representation of frequency content |
| **3D STL model** | Tactile   | Blind / low-vision users can *feel* the data as a 3D-printed terrain |

The goal is that **no single sense is required** to experience the data.
A deaf researcher can hold the 3D print. A blind researcher can listen to the
sonification. A sighted researcher can read the spectrogram. Everyone gets
the same underlying dataset.

---

## What It Does

1. **Loads real LIGO HDF5 data** from the Gravitational Wave Open Science Center
   (GWOSC). These are actual strain measurements from the H1 and L1 detectors.

2. **Processes the signal** — bandpass filtering, normalization, and a "data spread"
   control that ranges from melodic (compressed, tonal) to raw (full dynamic range).

3. **Three simultaneous output panels:**
   - **Waveform** — raw strain amplitude over time
   - **Spectrogram** — frequency content visualized with an inferno colormap,
     with a live playback cursor
   - **3D terrain preview** — isometric wireframe of the spectrogram surface,
     exportable as a 3D-printable STL file

4. **Real-time audio playback** with interactive controls:
   - Data Spread (melodic ↔ raw)
   - Volume
   - Speed / Pitch (higher speed = higher pitch — like speeding up a record)
   - Reverb
   - Bandpass low & high cutoff

5. **STL export** — one-click export of the spectrogram as a solid 3D mesh
   (terrain top + flat base + side walls) ready for slicing and 3D printing.

---

## Datasets

| File | Detector | Run | Sample Rate | Duration |
|------|----------|-----|-------------|----------|
| `H-H1_GWOSC_O4a_4KHZ_R1-1368195072-4096.hdf5` | H1 (Hanford) | O4a | 4096 Hz | ~68 min |
| `H-H1_GWOSC_O4a_4KHZ_R1-1368424448-4096.hdf5` | H1 (Hanford) | O4a | 4096 Hz | ~68 min |

Two datasets from the same detector and observing run but different GPS time
segments, allowing comparison of how different stretches of real data sound and
look when sonified.

---

## Known Issues & Next Steps

### Audio feedback (v0.2 findings)
- **Speed slider**: Was not live-updating pitch during playback — fixed with a
  pyo `SigTo` signal so pitch now tracks the slider in real time.
- **Bandpass filter**: Low-pass and high-pass adjustments were *barely audible*
  at certain settings. The filter range may need widening or the Q factor needs
  tuning for more dramatic effect.
- **Reverb/delay**: Effect was subtle. May need more aggressive wet/dry mix or
  additional delay/echo chains for spatial audio cues (important for users with
  hearing aids who rely on spatial separation).

### Planned improvements
- [ ] Live bandpass filter update (currently only on slider release)
- [ ] Add delay/echo effect separate from reverb
- [ ] Widen bandpass frequency range for more audible impact
- [ ] Support L1 (Livingston) detector files for cross-detector comparison
- [ ] Haptic feedback integration (gamepad rumble mapped to amplitude)
- [ ] Accessible keyboard navigation for all controls
- [ ] Screen reader annotations for panel content
- [ ] Audio description mode (spoken narration of data features)

---

## Tech Stack

- **Python 3.12** — main runtime
- **pyo** — real-time audio synthesis and DSP
- **pygame** — GUI rendering (lightweight, no heavy framework)
- **h5py** — LIGO HDF5 data loading
- **numpy** — signal processing (FFT, bandpass, normalization)
- **numpy-stl** — 3D mesh generation for STL export

---

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Run with default dataset (60s segment)
python src/ligo_sonifier.py

# Run with specific file and duration
python src/ligo_sonifier.py ~/Downloads/H-H1_GWOSC_O4a_4KHZ_R1-1368424448-4096.hdf5 --duration 120

# Keyboard shortcuts
#   SPACE  — play / stop
#   R      — reprocess audio with current slider values
#   E      — export STL file
#   Q/ESC  — quit
```

---

## License

Research / educational use. LIGO data is provided by GWOSC under their
[data use terms](https://gwosc.org/terms/).
