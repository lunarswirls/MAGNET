# Multi-Axis Gradiometric Noise Elimination and Tracking (MAGNET) Algorithm
An object that produces a magnetic field through intrinsic or induced means will affect the constant distribution of an 
ambient field, creating an abnormality referred to as a magnetic anomaly. Magnetic anomaly detection refers to the 
technique of determining characteristics of a magnetic target through the acquisition and processing of magnetic field 
information. Magnetic anomaly detection technology is used on Earth in fields such as aeromagnetic survey, navigation 
and positioning, and resource prospecting. This technology has been adapted for magnetic anomaly detection on a 
space-borne system, for source identification and tracking to reduce noise and recover the true ambient field. 

The first space mission to use the Multi-Axis Gradiometric Noise Elimination and Tracking (MAGNET) Algorithm will be
Lunar Vertex, a lunar lander and rover carrying magnetometers to explore a region on the surface known as a lunar swirl.


## Installation
MAGNET can be installed as a Python package called "magnetpy" for easy access to functions and visualization tools. 
This was done so that one could organize files into separate directories and still be able to reference each file
by defining a path relative to the package root name.

To install as a local branch (highly recommended. NOTE the "-e" flag to install as a local branch! Again, highly 
recommended you do not ignore the "-e" flag!)

```
pip install -e <path/to/your/local/python/package/directory containing setup.py>
```
NOTE: This is not a path to setup.py! This is the path to the root directory containing setup.py!

As an example, if you are user tycho and you want to use your copy of the magnetpy package that you put in 
`/homes/tycho/code/MAGNET/magnetpy`, you would do:
```
pip install -e /homes/tycho/code/MAGNET/
```


## Description
This is the noise removal and source tracking toolkit developed for the Lunar Vertex mission. More details about the 
mission and instrument suites are available in [TBD]().


## Visuals
Example of noise cleaning of simulation data from [Waller et al. 2025 (LPSC)](https://www.hou.usra.edu/meetings/lpsc2025/pdf/2592.pdf).

### Near-source non-dipolar test

The host-platform test superposes three simultaneous sources with different moment directions at distances comparable
to the array baseline. This produces a field with a large single-dipole model misfit and tests whether repeated MAGNET
corrections can recover the ambient field despite the non-dipolar interference.

Run the test from the repository root with:

```bash
python -m magnetpy.paper_figs.near_source_nondipolar_test_Fig9
```

The field generator is available as `generate_near_source_nondipolar_field()` in `Field_Utils.py`, and the runner writes
the source parameters, correction metrics, corrected time series, and summary figure to
`example_data/near_source_nondipolar_test/`.


## Authors and acknowledgment
Primary Author: [Dany Waller](danywaller.github.io)
- Email: [dany.c.waller@gmail.com](mailto:dany.c.waller@gmail.com)
