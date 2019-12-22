Module muvi.cine
================

Functions
---------

    
`T64_F(x)`
:   

    
`T64_F_ms(x)`
:   

    
`T64_S(s)`
:   

    
`sixteen2ten(b)`
:   

    
`sixteen2twelve(b)`
:   

    
`ten2sixteen(a)`
:   

    
`twelve2sixteen(a)`
:   

Classes
-------

`Cine(fn)`
:   Class for reading Vision Research CINE files, e.g. from Phantom cameras.
    
    Supports indexing, so frame can be accessed like list items, and ``len``
    returns the number of frames.  Iteration is also supported.
    
    Cine objects also use locks for file reading, allowing cine objects to
    be shared safely by several threads.
    
    Parameters
    ---------
    filename : string
        Source filename

    ### Instance variables

    `hash`
    :

    `trigger_time_p`
    :   Returns the time of the trigger, tuple of (datatime_object, fraction_in_ns)

    ### Methods

    `close(self)`
    :   Closes the cine file.

    `gamma_corrected_frame(self, frame_number, bottom_clip=0, top_clip=None, gamma=2.2)`
    :   Return a frame as a gamma corrected 'u1' array, suitable for saving
        to a standard image.
        
        Output is equal to: ``255 * ((original - bottom_clip) / (top_clip - bottom_clip))**(1/gamma)``
        
        Parameters
        ----------
        frame_number : integer
        gamma : float (default: 2.2)
            The gamma correction to apply
        top_clip : integer (default: 0)
        bottom_clip : integer (default: 2**real_bpp)
        
        Returns
        -------
        frame : numpy array (dtype='u1')

    `get_fps(self)`
    :   Get the frames per second of the movie.
        
        Returns
        -------
        fps : int

    `get_frame(self, frame_number)`
    :   Get a frame from the cine file.
        
        Parameters
        ----------
        frame_number : integer
        
        Returns
        -------
        frame : numpy array (dtype='u1' or 'u2', depending on bit depth)

    `get_time(self, frame_number)`
    :   Get the time of a specific frame.
        
        Parameters
        ----------
        frame_number : integer
        
        Returns
        -------
        time : float
            Time from start in seconds.

    `len(self)`
    :

    `next(self)`
    :

    `read_header(self, fields, offset=0)`
    :

    `read_tagged_blocks(self)`
    :

    `unpack(self, fs, offset=None)`
    :