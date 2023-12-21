We have seperate directories for the 2 databases we used and any other files related to them, such as the code used for filtering. For the Meteoretical Bullitin database there is a folder called pages which contain the tables (in csv format) extracted from the pages on the MB database website using the following settings:
- Display: Show decimal degrees, Sort by mass, 5000 lines/pg, Normal table
- Limit to approved meteorite names: checked
(the other settings are kept at the default), link website: https://www.lpi.usra.edu/meteor/metbull.php

The NASA database we got from kaggle, link: https://www.kaggle.com/datasets/ulrikthygepedersen/meteorite-landings 

There is also a folder "land" which contains files that are used to calculate the land ratio for each latitude and longitude in 'latslongs.py'. These values can be found in the 'landratio.xlxs' file, 

hypothesis2mass,  hypothesis2loc and hypothesis2class (which is an extra) create the results and plots from the RQ "Are there differences between fallen and found meteorites"
project.py and final.ipynb are used for the RQ "Is there periodicity in the amount of fallen meteorites?" The first file has the code for autocorrelation and the second for the Fourier analysis. The codes for the cosine and uniform distribution for the RQ "Is there bias in terms of location for the meteorites?" can also be found in the final.ipynb
hypothesis4.py is used for the question "Are there trends in the location of fallen meteorites?"
