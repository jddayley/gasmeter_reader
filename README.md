# Gas Meter Reader

I started from David Padbury's [Power Meter Reader](https://github.com/davidpadbury/power-meter-reader) but made fairly extensive modifications to get it to read my gas meter. My gas company is Eversource in Hopkinton, MA.

Steps:
1. Crop down to the meaningful dials on my meter. The crop is specific to my meter and camera position.  This can be changed in gas_meter_read.py at line 185.
2. Run an aggressive normalization to squash some blacks & whites
3. Use HoughCircles to find the dials
4. Use Canny to do edge detection on a slightly blurred image
5. Detect the angle of the dials by brute-force looking for the cleanest line (fewest edges) from the center to the edge of the dial. The needles are too stubby to use line detection for this purpose.
6. Read the final dial fractionally. The final dial's integer represents 1000 cu. ft. of gas which is not a not very fine grained.


# Hardware

- I am using a Wyze v3 USB webcam.   It would not focus close enough so I followed the youtube instructions that allowed me  (https://www.youtube.com/watch?v=PnqDFVH_lfU&t=367s) to adjust the lense and focus on the gas meter.
- For lighting, I'm using the LED lighting. https://www.amazon.com/gp/product/B072QWJRBS/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&th=1
- I've enclosed everything in a Register Duck Boot. https://www.lowes.com/pd/IMPERIAL-10-in-x-6-in-x-6-in-Galvanized-Steel-Straight-Register-Duct-Boot/1000237469

# data errors
Anecdotally, the occasional errors seem to be down to the size & position of the detected circles. So, do some things to combat:
1. Take a number of consecutive frames prior to analysis (5)
1. Compute the median of the x, y, and radius values of detected circles
1. Use that center & radius for analysis of all frames
1. Take the median of each digit (and the fraction) as computed for each frame

# Next steps
I am working on a controlled environment.  I purchased a metal enclosure and led lighting.  This will elimate the data errors and improve the reliability. 
