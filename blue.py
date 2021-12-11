from bluepy import btle
import sys, binascii, struct
from bluepy.btle import UUID, Peripheral
print ("Connecting...")
#Humidifier
address = "24:d7:eb:01:af:02".upper()
#address = "c6:0e:6f:d8:cd:fc".upper()
#address = "C2:7B:7C:49:CB:06"
p = btle.Peripheral(address)
print( "Services...")
for svc in p.services:
    print( str(svc))
print( "Get Name...")
dev_name_uuid = UUID(0x2A00)
ch = p.getCharacteristics(uuid=dev_name_uuid)[0]
if (ch.supportsRead()):
     print(ch.read())
print( "Get Descriptors...")
descriptors=p.getDescriptors(1,0x00F) #Bug if no limt is specified the function wil hang 
                                      # (go in a endless loop and not return anything)
print("UUID                                  Handle UUID by name")
for descriptor in descriptors:
   print ( " "+ str(descriptor.uuid) + "  0x" + format(descriptor.handle,"02X") +"   "+ str(descriptor) )
print( "Get Characteristics...")
chList = p.getCharacteristics()
print("Handle   UUID                                Properties")
print("-------------------------------------------------------"  )                      
#d = map(chList[2].read())
#print( str(list(d)) )
for ch in chList:
   print ("  0x"+ format(ch.getHandle(),'02X')  +"   "+str(ch.uuid) +" " + ch.propertiesToString())
  # print(str(ch.read()))
   if (ch.supportsRead()):
       print("Data: " + str(ch.read()))
       try:
          val = ch.read() 
          val = binascii.b2a_hex(val)
          val = binascii.unhexlify(val)
          #val = struct.unpack('f', val)[0]
          print( str(val) + " deg C")
       except:
           print("error")
       #print(struct.unpack('<f',ch.read()[2:6])[0])