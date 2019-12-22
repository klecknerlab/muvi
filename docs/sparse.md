Module muvi.sparse
==================

Functions
---------

    
`c_unpack(s, array_size, dt)`
:   

    
`eval_slice(s, N)`
:   

    
`if_int(str)`
:   

    
`make_block(a, sparse=None)`
:   

    
`pack_func(m)`
:   

    
`sparse_decode(s, shape, dtype)`
:   

    
`sparse_encode(s)`
:   

    
`unpack_block(s)`
:   

    
`unpack_func(m)`
:   

Classes
-------

`Sparse(fn, read_write='r', header_dict=None, max_blocks=1024, cache_blocks=False, preload=False)`
:   

    ### Descendants

    * muvi.sparse.Sparse4D

    ### Class variables

    `file_desc`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    `file_id`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    `post_comments`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    ### Methods

    `append_array(self, a, sparse=None)`
    :

    `append_block(self, block)`
    :

    `close(self)`
    :

    `get_raw_block(self, block_num)`
    :

    `read_block(self, block_num)`
    :

    `read_header_line(self)`
    :

    `read_struct(self, fmt)`
    :

    `unpack_block(self, desc, extra_header, data)`
    :

    `write_struct(self, fmt, *args)`
    :

`Sparse4D(fn, read_write='r', header_dict=None, max_blocks=1024, cache_blocks=False, preload=False)`
:   

    ### Ancestors (in MRO)

    * muvi.sparse.TextHeaderedBinary

    ### Methods

    `frame(self, n)`
    :

`TextHeaderedBinary(fn, read_write='r', header_dict=None, max_blocks=1024, cache_blocks=False, preload=False)`
:   

    ### Descendants

    * muvi.sparse.Sparse4D

    ### Class variables

    `file_desc`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    `file_id`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    `post_comments`
    :   str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str
        
        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to sys.getdefaultencoding().
        errors defaults to 'strict'.

    ### Methods

    `append_array(self, a, sparse=None)`
    :

    `append_block(self, block)`
    :

    `close(self)`
    :

    `get_raw_block(self, block_num)`
    :

    `read_block(self, block_num)`
    :

    `read_header_line(self)`
    :

    `read_struct(self, fmt)`
    :

    `unpack_block(self, desc, extra_header, data)`
    :

    `write_struct(self, fmt, *args)`
    :