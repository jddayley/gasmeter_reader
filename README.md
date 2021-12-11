# Gas Meter Reader

I started from David Padbury's [Power Meter Reader](https://github.com/davidpadbury/power-meter-reader) but made fairly extensive modifications to get it to read my gas meter. My gas company is National Grid in Newton, MA.

Steps:
1. Crop down to the meaningful dials on my meter. The crop is specific to my meter and camera position.
2. Run an aggressive normalization to squash some blacks & whites
3. Use HoughCircles to find the dials, like the original
4. Use Canny to do edge detection on a slightly blurred image, like the original
5. Detect the angle of the dials by brute-force looking for the cleanest line (fewest edges) from the center to the edge of the dial. The needles are too stubby to use line detection for this purpose.
6. Read the final dial fractionally. The final dial's integer represents 1000 cu. ft. of gas which is not a not very fine grained.

I kept most of the "dodgy Python" code from the upstream project because it works, and my own Python skills are not much better.

# Hardware

I am using a nearly-antique USB webcam, with a max resolution of 1280x1024. It can't focus close enough so I attached a +6-diopter filter in front of the lens.
For lighting, the meter is rather annoying because its case is highly reflective and slightly curved, and the needles inside are shiny black plastic as well. So reflections are a real issue. I used a pair of gooseneck USB LED lights, two of them to avoid harsh shadows. I also needed to add a baffle to prevent bad reflections from room light and a basement window.

I attached a 4-port USB hub and the camera to a stick of wood suspended from a ceiling joist in the basement. So the whole thing is low (USB) voltage and does not actually touch the meter in any way.

# data errors
Anecdotally, the occasional errors seem to be down to the size & position of the detected circles. So, do some things to combat:
1. Take a number of consecutive frames prior to analysis (5)
1. Compute the median of the x, y, and radius values of detected circles
1. Use that center & radius for analysis of all frames
1. Take the median of each digit (and the fraction) as computed for each frame

Medians have helped, but I am still getting some errors. I think I need to abandon using Canny, and instead threshold the meter image and look for the angle with the least number of bright pixels along the line.
-> So far this seems to be working better.
