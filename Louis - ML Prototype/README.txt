Steps I took to get my Prototype to work
Download Latest Version of Python (3.7.1 as of 11/27/2018) (download here: https://www.python.org/downloads/release/python-371/)
Customize Installation
	Install for all users
	Associate files with python (requires the py launcher)
	Create shortcuts for installed applications
	Add python to environment variables
	Precompile standard library
		Disable path length limit after installation
			To see if everything worked, open cmd via Shift+Right Click -> Open Windows PowerShell here, type python
			
Download Visual C++ 2017 Redistributable

pip install "numpy-1.14.6+mkl-cp37-cp37m-win32.whl" (download here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)

pip install "opencv_python-3.4.4+contrib-cp37-cp37m-win32.whl" (download here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)

How to use ml_prototype.py
Open ml_prototype.py
Press b to capture background model
Press r to reset the background model
Press Esc to exit
