# Define a variable for classpath 
CLASS_PATH = ../bin 

# Define a virtual path for .class in the bin directory 
vpath %.class $(CLASS_PATH) 

# $* matches the target filename without the extension 
HOGDescriptor.h : HOGDescriptor.class javah -classpath $(CLASS_PATH) $*