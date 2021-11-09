# Gas Meter Reader

I started from David Padbury's [Power Meter Reader](https://github.com/davidpadbury/power-meter-reader) but made fairly extensive modifications to get it to read my gas meter. My gas company is National Grid in Hopkinton, MA.

Steps:
1. Crop down to the meaningful dials on my meter. The crop is specific to my meter and camera position.
2. Run an aggressive normalization to squash some blacks & whites
3. Use HoughCircles to find the dials, like the original
4. Use Canny to do edge detection on a slightly blurred image, like the original
5. Detect the angle of the dials by brute-force looking for the cleanest line (fewest edges) from the center to the edge of the dial. The needles are too stubby to use line detection for this purpose.
6. Read the final dial fractionally. The final dial's integer represents 1000 cu. ft. of gas which is not a not very fine grained.

I kept most of the "dodgy Python" code from the upstream project because it works, and my own Python skills are not much better.

# Hardware

I am using a Wyze v3 USB webcam.   It would not focus close enough so I followed the youtube instructions (https://www.youtube.com/watch?v=PnqDFVH_lfU&t=367s) and was able to adjust the lense.

# data errors
Anecdotally, the occasional errors seem to be down to the size & position of the detected circles. So, do some things to combat:
1. Take a number of consecutive frames prior to analysis (5)
1. Compute the median of the x, y, and radius values of detected circles
1. Use that center & radius for analysis of all frames
1. Take the median of each digit (and the fraction) as computed for each frame

Medians have helped, but I am still getting some errors. I think I need to abandon using Canny, and instead threshold the meter image and look for the angle with the least number of bright pixels along the line.
-> So far this seems to be working better.
