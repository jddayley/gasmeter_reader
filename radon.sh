#!/bin/bash
x=1
s=1
while [ $s -le 5 ]
do
  echo "# of Radon $x samples"
  x=$(( $x + 1 ))
  python3 radon_meter.py -v -a C2:7B:7C:49:CB:06 -m -ma -ms 192.168.0.116 -mp 1883 -mu jddayley -mw java
  sleep 300
done
