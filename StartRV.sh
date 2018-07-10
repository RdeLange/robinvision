#!/bin/bash

echo "RobinVision Docker Container Starting." 
/etc/init.d/apache2 start
echo "RobinVision Image Management Starting." 
python RobinVision.py
echo "RobinVision API Starting." 
echo "RobinVision Docker Container Now Running." 


