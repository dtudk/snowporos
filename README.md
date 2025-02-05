We present this code based on existing functions present at PoreSpy.https://porespy.org/

The code segments the interconnected pores of a certain material into different units. 

The function can be found in the file snow_poros.py


Installation

Installation of this code should follow these instructions

./configure --prefix=<path-to-destination>
make
make install


Usage

The actual code that performs the steps of the SNOWPOROS algorithm can be found in the file snow_poros.py

An real example that segments a binary image (image3_.tiff)is illustrated in the file example_using_snowporos.py. 

License
This code is released under the MIT License, see more in license_terms.txt.
